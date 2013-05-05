$(document).ready(function(){

    $("#capture-face").hide();
    $("#capture-result").hide();

    // get DOM elements
    var video = $("#live").get()[0];
    var canvas = $("#canvas");
    var ctx = canvas.get()[0].getContext('2d');
    var localMediaStream = null;

    // mirror canvas
    ctx.translate(Eigenfaces.WIDTH, 0);
    ctx.scale(-1, 1);

    // create websocket
    var ws = new WebSocket("ws://" + Eigenfaces.HOST + ":" +
                           Eigenfaces.PORT + "/websocket");
    ws.onopen = function() {
        console.log("Opened connection to websocket");
    };

    ws.onmessage = function(msg) {
        // display updated notification on if there is a face or not
        document.getElementById('changetext').innerHTML = msg.data;
    };

    ws.onclose = function(msg) {
        console.log("Closed connection to websocket");
    };

    $('#capture-button').click( function() {
        $("#capture-button").hide();

        navigator.webkitGetUserMedia(
	{video: true, audio: false},
	    function(stream) {
	        video.src = webkitURL.createObjectURL(stream);
                localMediaStream = stream;
		// capture input for 10 seconds

		setTimeout (
		    function () {
		        $("#capture-face").show();
		    }, 10000);

		for (i = 0; i <= 10000; i+=200) {
		
		    setTimeout(
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
		// capture input for 10 seconds

		setTimeout (
		    function () {
		        $("#capture-result").show();
		    }, 10000);

		for (i = 0; i < 10000; i+=200) {
		
		    setTimeout(
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

    $('#capture-stream').click( function() {
        $("#capture-stream").hide();
        setInterval(
            // capture a constant stream
            function () {
	        ctx.drawImage(video, 0, 0, Eigenfaces.WIDTH, Eigenfaces.HEIGHT);
	        var data = canvas.get()[0].toDataURL('image/jpeg', 1.0);
	        ws.send(data.split(',')[1]);
            }, 200);
    });

    $('#capture-snapshot').click( function() {
        if (localMediaStream) {
            ctx.drawImage(video,0,0, Eigenfaces.WIDTH, Eigenfaces.HEIGHT);
	    var data = canvas.get()[0].toDataURL('image/jpeg', 1.0);
	    ws.send(data.split(',')[1]);
        }
    });

    // close socket when the document closes
    $(document).unload(function() {
        ws.close();
    });

});
