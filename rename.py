import os
os.chdir("\Dataset\Val\Vadiveluu")
i=1
for file in os.listdir():
    src=file
    dst="vadiveluu"+str(i)+".jpg"
    os.rename(src,dst)
    i+=1
    