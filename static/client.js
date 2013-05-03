$(document).ready(function(){
    // Google Maps setup
    $("#capture-face").hide();
    $("#capture-result").hide();

    // variables for the DOM elements
    var video = $("#live").get()[0];
    var canvas = $("#canvas");
    var ctx = canvas.get()[0].getContext('2d');
    var localMediaStream = null;
    // mirror the canvas
    ctx.translate(Eigenfaces.WIDTH, 0);
    ctx.scale(-1, 1);

    // interval variable to send a constant stream to the server
    var timer;

    // establish websocket
    var ws = new WebSocket("ws://" + Eigenfaces.HOST + ":" +
                           Eigenfaces.PORT + "/websocket");
    ws.onopen = function() {
        console.log("Opened connection to websocket");
    };
    ws.onmessage = function(msg) {
        // display the updated frame
        var target = document.getElementById("target");
        url = window.webkitURL.createObjectURL(msg.data.slice(10, msg.data.size));
        target.onload = function() {
            window.webkitURL.revokeObjectURL(url);
        };
        target.src = url;
    };
    ws.onclose = function(msg) {
        window.clearInterval(timer);
        console.log("Closed connection to websocket");
    };

    $('#capture-button').click( function() {
        $("#capture-button").hide();

        navigator.webkitGetUserMedia(
	{video: true, audio: false},
	    function(stream) {
	        video.src = webkitURL.createObjectURL(stream);
                localMediaStream = stream;
		// send a constant stream to the server

		setTimeout (
		    function () {
		        $("#capture-face").show();
		    }, 10000);

		for (i = 0; i <= 10000; i+=200) {
		
		    timer = setTimeout(
		        function () {
			    ctx.drawImage(video, 0, 0, Eigenfaces.WIDTH, Eigenfaces.HEIGHT);
			    var data = canvas.get()[0].toDataURL('image/jpeg', 1.0);
			    ws.send(data.split(',')[1]);
	       	        }, i);
                }


	    },
	    function(err) {
		ws.close();
		console.log("Unable to get video stream!, Error: " + err)
            }
	);
    });

    $('#capture-face').click( function() {
        $("#capture-face").hide();
        navigator.webkitGetUserMedia(
	{video: true, audio: false},
	    function(stream) {
	        video.src = webkitURL.createObjectURL(stream);
                localMediaStream = stream;
		// send a constant stream to the server

		setTimeout (
		    function () {
		        $("#capture-result").show();
		    }, 10000);

		for (i = 0; i < 10000; i+=200) {
		
		    timer = setTimeout(
		        function () {
			    ctx.drawImage(video, 0, 0, Eigenfaces.WIDTH, Eigenfaces.HEIGHT);
			    var data = canvas.get()[0].toDataURL('image/jpeg', 1.0);
			    ws.send(data.split(',')[1]);
	       	        }, i);
                }

                //setTimeout (
                //    function () {
                //        document.getElementById('canvas').style.display = 'block';
                //}, 10500);
	    },
	    function(err) {
		ws.close();
		console.log("Unable to get video stream!, Error: " + err)
            }
	);
    });

    $('#capture-stream').click( function() {
        $("#capture-stream").hide();
        setInterval(
            function () {
	        ctx.drawImage(video, 0, 0, Eigenfaces.WIDTH, Eigenfaces.HEIGHT);
            }, 200);
    });

    $('#capture-snapshot').click( function() {
        if (localMediaStream) {
            ctx.drawImage(video,0,0, Eigenfaces.WIDTH, Eigenfaces.HEIGHT);
	    var data = canvas.get()[0].toDataURL('image/jpeg', 1.0);
	    ws.send(data.split(',')[1]);
            //document.querySelector('img')src = canvas.toDataURL('image/webp');
        }
    });

    // close socket when the document closes
    $(document).unload(function() {
        ws.close();
    });

});
