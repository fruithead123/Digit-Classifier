// Brush colour and size
const colour = "black";
const strokeWidth = 25;

// Drawing state
let latestPoint;
let drawing = false;

// Set up our drawing context
const canvas = document.getElementById("canvas");
const context = canvas.getContext("2d");

canvas.height = 280;
canvas.width = canvas.height;


function resetCanvas(){
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "white";
    context.fillRect(0, 0, canvas.width, canvas.height);
}

// Drawing functions

const continueStroke = newPoint => {
    context.beginPath();
    context.moveTo(latestPoint[0], latestPoint[1]);
    context.strokeStyle = colour;
    context.lineWidth = strokeWidth;
    context.lineCap = "round";
    context.lineJoin = "round";
    context.lineTo(newPoint[0], newPoint[1]);
    context.stroke();

    latestPoint = newPoint;
};

// Event helpers

const startStroke = point => {
    drawing = true;
    latestPoint = point;
};

const BUTTON = 0b01;
const mouseButtonIsDown = buttons => (BUTTON & buttons) === BUTTON;

// Event handlers

const mouseMove = evt => {
    if (!drawing) {
        return;
    }
    continueStroke([evt.offsetX, evt.offsetY]);
};

const mouseDown = evt => {
    if (drawing) {
        return;
    }
    evt.preventDefault();
    canvas.addEventListener("mousemove", mouseMove, false);
    startStroke([evt.offsetX, evt.offsetY]);
};

const mouseEnter = evt => {
    if (!mouseButtonIsDown(evt.buttons) || drawing) {
        return;
    }
    mouseDown(evt);
};

const endStroke = evt => {
    if (!drawing) {
        return;
    }
    drawing = false;
    evt.currentTarget.removeEventListener("mousemove", mouseMove, false);
};

// Register event handlers

canvas.addEventListener("mousedown", mouseDown, false);
canvas.addEventListener("mouseup", endStroke, false);
canvas.addEventListener("mouseout", endStroke, false);
canvas.addEventListener("mouseenter", mouseEnter, false);

const getTouchPoint = evt => {
    if (!evt.currentTarget) {
        return [0, 0];
    }
    const rect = evt.currentTarget.getBoundingClientRect();
    const touch = evt.targetTouches[0];
    return [touch.clientX - rect.left, touch.clientY - rect.top];
};

const touchStart = evt => {
    if (drawing) {
        return;
    }
    evt.preventDefault();
    startStroke(getTouchPoint(evt));
};

const touchMove = evt => {
    if (!drawing) {
        return;
    }
    continueStroke(getTouchPoint(evt));
};

const touchEnd = evt => {
    drawing = false;
};

canvas.addEventListener("touchstart", touchStart, false);
canvas.addEventListener("touchend", touchEnd, false);
canvas.addEventListener("touchcancel", touchEnd, false);
canvas.addEventListener("touchmove", touchMove, false);


//Update results

function rgb(r, g, b){
    return "rgb("+r+","+g+","+b+")";
}

function predict(){
    var data = canvas.toDataURL();
    $.ajax({
        type: "POST",
        url: '/digitPredict',
        data: JSON.stringify({"imageBase64": data}),
        contentType: 'application/json',
        success: function(response){
            //console.log(response)
            document.getElementById("digitPredictionValue").textContent = response['winner']

            var dig;
            for(dig=0; dig<10; dig++){
                document.getElementById("digit" + dig + "top").style.height = (280 - Math.max(1,response['pred'][dig] * 280)) + "px";
                document.getElementById("digit" + dig + "bot").style.height = Math.max(1, response['pred'][dig] * 280) + "px";
                document.getElementById("digit" + dig + "bot").style.backgroundColor = rgb(2*255*(1-response['pred'][dig]), 2*255*response['pred'][dig],0);
            }
        },
    });
}


$(document).ready(function(){
    resetCanvas();
    //$("#digitSubmit").click(predict);
    setInterval(predict, 100);
});
