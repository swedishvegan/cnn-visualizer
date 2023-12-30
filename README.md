# cnn-visualizer
A tool to help visualize how CNNs work behind the scenes

## About this project

__Please check out the PDF in the docs folder for details on how this project works.__

This project is a simple tool that lets you visualize the types of patterns that activate specific kernels in a convolutional neural network. It was designed as an educational tool to help demystify the way CNNS work and what it is they actually learn to recognize. This particular program uses the GoogLeNet model, although it should not be too difficult to play with the source code to make it work with different models.

With this program, you can specify which kernel you want to visualize, as well as several other parameters that let you customize the output image style. Then the program uses PyTorch to generate and output an image, using GPU acceleration if applicable.

This was my final project for my data science class in spring 2023, but I thought it was interesting enough to post on my GitHub.

