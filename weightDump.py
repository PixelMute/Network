model = NULL # Get your model however
param = model.named_parameters()
f = open("weightdump.txt", "w")

# Save floats
for x in param:
    print("Name:" + x[0] + ". shape: " + str(x[1].data.shape))
    if (len(x[1].data.shape) == 3):
        for i in range(len(x[1].data)):
            for j in range(len(x[1].data[i])):
                for k in range(len(x[1].data[i][j])):
                    f.write(str(f'{x[1].data[i][j][k]:.8f}')) # You can change the 8 for higher/lower precision. Too high of a value, and it doesn't work well though.
                    if ((i + 1) * (j+1) * (k+1)) != x[1].data.shape[0] * x[1].data.shape[1] * x[1].data.shape[2]:
                        f.write(" ")
    elif (len(x[1].data.shape) == 2):
        for i in range(len(x[1].data)):
            for j in range(len(x[1].data[i])):
                f.write(str(f'{x[1].data[i][j]:.8f}'))
                if ((i + 1) * (j+1)) != x[1].data.shape[0] * x[1].data.shape[1]:
                        f.write(" ")
    elif (len(x[1].data.shape) == 1):
        for i in range(len(x[1].data)):
            f.write(str(f'{x[1].data[i]:.8f}'))
            if ((i + 1)) != x[1].data.shape[0]:
                f.write(" ")
    f.write("\n")
f.close()
