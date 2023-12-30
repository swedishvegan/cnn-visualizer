layer = input("Layer name: ")
n = int(input("# Features: "))
print("Generating " + layer + ".bat...")
with open(layer + ".bat", "w") as f: 
    for i in range(n): f.write("python image_gen.py --octaves 3 --size 128 --outscale 4.0 --layer " + layer + " --feature " + str(i) + "\n")

    