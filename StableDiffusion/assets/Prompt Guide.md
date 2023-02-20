Basics of Prompt Engineering

A guide by Graverman

Today I propose a simple formula for beginners to use and create better generations with text to image AI. This was tested on stable diffusion but it should work on any model if it was trained on enough art data.

After reading this document and applying these simple steps, you’ll be able to generate better images with the same amount of effort.


1. Raw prompt

Raw prompt is the simplest way of describing what you want to generate, for instance;

Panda
A warrior with a sword
Skeleton

This is the basic building block of any prompt. Most new people start by only using raw prompts, this is usually a mistake as the images you generate like this tend to get random and chaotic. Here are some examples that I generated with running the earlier prompts

As you can see, these images have random scenery and don’t look very aesthetically pleasing, I definitely wouldn’t consider them art. This brings me to my next point;


2. Style

Style is a crucial part of the prompt. The AI, when missing a specified style, usually chooses the one it has seen the most in related images, for example, if I generated landscape, it would probably generate realistic or oil painting looking images. Having a well chosen style + raw prompt is sometimes enough, as the style influences the image the most right after the raw prompt.

The most commonly used styles include:

Realistic
Oil painting
Pencil drawing
Concept art

I’ll examine them one by one to give an overview on how you might use these styles.

In the case of a realistic image, there are various ways of making it the style, most resulting in similar images. Here are some commonly used techniques of making the image realistic:

a photo of + raw prompt
a photograph of + raw prompt
raw prompt, hyperrealistic
raw prompt, realistic
You can of course combine these to get more and more realistic images.

To get oil painting you can just simply add “an oil painting of” to your prompt. This sometimes results in the image showing an oil painting in a frame, to fix this you can just re-run the prompt or use raw prompt + “oil painting”

To make a pencil drawing just simply add “a pencil drawing of” to your raw prompt or make your prompt raw prompt + “pencil drawing”.

The same applies to landscape art.


3. Artist

To make your style more specific, or the image more coherent, you can use artists’ names in your prompt. For instance, if you want a very abstract image, you can add “made by Pablo Picasso” or just simply, “Picasso”.

Below are lists of artists in different styles that you can use, but I always encourage you to search for different artists as it is a cool way of discovering new art.

Portrait

John Singer Sargent
Edgar Degas
Paul Cézanne
Jan van Eyck
Oil painting

Leonardo DaVinci
Vincent Van Gogh
Johannes Vermeer
Rembrandt
Pencil/Pen drawing

Albrecht Dürer
Leonardo da Vinci
Michelangelo
Jean-Auguste-Dominique Ingres
Landscape art

Thomas Moran
Claude Monet
Alfred Bierstadt
Frederic Edwin Church
Mixing the artists is highly encouraged, as it can lead to interesting-looking art.


4. Finishing touches

This is the part that some people take to extremes, leading to longer prompts than this article. Finishing touches are the final things that you add to your prompt to make it look like you want. For instance, if you want to make your image more artistic, add “trending on artstation”. If you want to add more realistic lighting add “Unreal Engine.” You can add anything you want, but here are some examples:

Highly detailed, surrealism, trending on art station, triadic color scheme, smooth, sharp focus, matte, elegant, the most beautiful image ever seen, illustration, digital paint, dark, gloomy, octane render, 8k, 4k, washed colors, sharp, dramatic lighting, beautiful, post processing, picture of the day, ambient lighting, epic composition


5. Conclusion

Prompt engineering allows you to have better control of what the image will look like. It (if done right) improves the image quality by a lot in every aspect. If you enjoyed this “article”, well, I’m glad I didn’t waste my time. If you see any ways that I can improve this, definitely let me know on discord