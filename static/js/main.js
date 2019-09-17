$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();
    let net;
    let img = document.createElement('img');
    const webcamElement = document.getElementById('webcam');
    const classifier = knnClassifier.create();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').attr('src', e.target.result);
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);

                //couldn't find an other way..
                img.src = e.target.result;
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(async function () {
        /*var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#result').text(' Result:  ' + data);
                console.log('Success!');
            },
        });*/

        // Show loading animation
        $(this).hide();
        $('.loader').show();
        let image = img;

        net = await mobilenet.load();
        console.log('Successfully loaded model');

        
        

        /*console.log('before load, complete=' + image.complete);
        image.onload = function() {
            console.log('after load, complete=' + image.complete);
        }*/
        const result = await net.classify(image);
     
        let prop = result["0"].probability.toFixed(5)

        $('.loader').hide();
        $('#result').fadeIn(600);
        $('#result').text(' Result:  ' + result["0"].className +',  probability: '+prop);
        console.log('Success!');

    });


    async function setupWebcam() {
        return new Promise((resolve, reject) => {
        const navigatorAny = navigator;
        navigator.getUserMedia = navigator.getUserMedia ||
            navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia ||
            navigatorAny.msGetUserMedia;
        if (navigator.getUserMedia) {
            navigator.getUserMedia({video: true},
            stream => {
                webcamElement.srcObject = stream;
                webcamElement.addEventListener('loadeddata',  () => resolve(), false);
            },
        error => reject());
    } else {
      reject();
    }
  });
}

    async function app() {
        console.log('Loading mobilenet..');

        // Load the model.
        net2 = await mobilenet.load();
        net3 = await mobilenet.load();
        console.log('Successfully loaded webcam model');
      
        await setupWebcam();
        // Reads an image from the webcam and associates it with a specific class
        // index.
        const addExample = classId => {
            // Get the intermediate activation of MobileNet 'conv_preds' and pass that
            // to the KNN classifier.
            const activation = net3.infer(webcamElement, 'conv_preds');
            // Pass the intermediate activation to the classifier.
            classifier.addExample(activation, classId);
        };

        // When clicking a button, add an example for that class.
        document.getElementById('class-a').addEventListener('click', () => addExample(0));
        document.getElementById('class-b').addEventListener('click', () => addExample(1));
        document.getElementById('class-c').addEventListener('click', () => addExample(2));

        while (true) {
        const liveresult = await net2.classify(webcamElement);
        $('#liveresult').text(`
             prediction: ${liveresult[0].className}\n
             probability: ${liveresult[0].probability.toFixed(5)}
            `);
        
        if (classifier.getNumClasses() > 0) {
            // Get the activation from mobilenet from the webcam.
            const activation = net3.infer(webcamElement, 'conv_preds');
            // Get the most likely class and confidences from the classifier module.
            const webresult = await classifier.predictClass(activation);

            const classes = ['A', 'B', 'C'];

        
            
            $('#classresult').text(`
              prediction: ${classes[webresult.classIndex]}\n
              probability: ${webresult.confidences[webresult.classIndex]}
            `)

        }
        // Give some breathing room by waiting for the next animation frame to
        // fire.
        await tf.nextFrame();
      }
    }
    app()


});
