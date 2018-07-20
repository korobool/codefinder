# codefinder
An ML (NN) based example of JavaScript code detector in a raw input.

## Repository content
1. *train.py* is a script for model training.
2. *model.py* is a library that describes the network architecture.
3. *helpers.py* is a library that performs padding with zeros and converts data to numpy arrays.
4. *code_detector.py* is a library containing the code-detector class, works on the pre-trained model.
5. *Code_Detector.ipynb* is a notebook demonstrating the *code_detector.py* example.

## How to use
```Python
from code_detector import Detector
detector = Detector()
```

```Python
file = open('for_test_01.txt', 'r')
text = file.read()
text_with_marks = detector.detect(text)

text_with_marks = detector.detect(text)
```


```
<-start_text->
       But it wasn't until that night that he could return to the
intricacies of the problem. Friend or foe? Christ, what had they been
discussing when Hrrula doodled in the dust? Oh yeah, about the colony
leaving because the planet was already inhabited. And then he'd gone on at
length about the long history of the Terranic aggression and genocide.
Ohhh, he groaned at the memory of such an admission reaching Hrruban ears;
ears unfamiliar with the Terran language. What on earth had possessed him
to talk about that phase of Terran history in the first place? What an
impression
 
 * 
 *   <img src="normal_image.png" data-ro
<-end_text->

<-start_code->
llover="rollover_image.png">
 * 
 * Note that this module requires onLoad.js
 */
onLoad(function() { // Everything in one anonymous function: no symbols defined
    // Loop through all images, looking for the data-rollover attribute
    for(var i = 0; i < document.images.length; i++) {
        var img = document.images[i]; 
        var rollover = img.getAttribute("data-rollover"); 


<-end_code->

<-start_text->
length about the long history of the Terranic aggression and genocide.
Ohhh, he groaned at the memory of such an admission reaching Hrruban ears;
ears unfamiliar with the Terran language. What on earth had possessed him
to talk about that phase of Terran history in the first place? What an
impression

<-end_text->
```

Implementation ihetits sequense to sequense implementation found [here](https://github.com/ematvey/tensorflow-seq2seq-tutorials/blob/master/2-seq2seq-advanced.ipynb)
