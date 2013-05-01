$(document).ready(function(){
    // Google Maps setup
    $("#capture-face").hide();

    // variables for the DOM elements
    var video = $("#live").get()[0];
    var canvas = $("#canvas");
    var ctx = canvas.get()[0].getContext('2d');

    // mirror the canvas
    ctx.translate(MotionApp.WIDTH, 0);
    ctx.scale(-1, 1);

    // interval variable to send a constant stream to the server
    var timer;

    // establish websocket
    var ws = new WebSocket("ws://" + MotionApp.HOST + ":" +
                           MotionApp.PORT + "/websocket");
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
		// send a constant stream to the server

		setTimeout (
		    function () {
		        $("#capture-face").show();
		    }, 10000);

		for (i = 0; i < 10000; i+=200) {
		
		    timer = setTimeout(
		        function () {
			    ctx.drawImage(video, 0, 0, MotionApp.WIDTH, MotionApp.HEIGHT);
			    var data = canvas.get()[0].toDataURL('image/jpeg', 1.0);
			    ws.send(data.split(',')[1]);
	       	        }, i);
                }

                setTimeout (
                    function () {
                        document.getElementById('canvas').style.display = 'block';
                }, 10500);
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
		// send a constant stream to the server

		for (i = 0; i < 10000; i+=200) {
		
		    timer = setTimeout(
		        function () {
			    ctx.drawImage(video, 0, 0, MotionApp.WIDTH, MotionApp.HEIGHT);
			    var data = canvas.get()[0].toDataURL('image/jpeg', 1.0);
			    ws.send(data.split(',')[1]);
	       	        }, i);
                }

                setTimeout (
                    function () {
                        document.getElementById('canvas').style.display = 'block';
                }, 10500);
	    },
	    function(err) {
		ws.close();
		console.log("Unable to get video stream!, Error: " + err)
            }
	);
    });

    // close socket when the document closes
    $(document).unload(function() {
        ws.close();
    });

});
