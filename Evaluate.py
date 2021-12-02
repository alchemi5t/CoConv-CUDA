import os

os.system("rm -f speedup");
os.system("touch speedup");

os.system("python3 datagen.py 5 5 3 4")
os.system("./coconv 5 5 3 4")
os.system("python3 pytorch.py 5 5 3 4")
os.system("rm -f in*");
os.system("rm -f weights*");

os.system("python3 datagen.py 16 16 12 8")
os.system("./coconv 16 16 12 8")
os.system("python3 pytorch.py 16 16 12 8")
os.system("rm -f in*");
os.system("rm -f weights*");

os.system("python3 datagen.py 32 32 32 8")
os.system("./coconv 32 32 32 8")
os.system("python3 pytorch.py 32 32 32 8")
os.system("rm -f in*");
os.system("rm -f weights*");

os.system("python3 datagen.py 64 64 32 12")
os.system("./coconv 64 64 32 12")
os.system("python3 pytorch.py 64 64 32 12")
os.system("rm -f in*");
os.system("rm -f weights*");

os.system("python3 datagen.py 64 64 32 16")
os.system("./coconv 64 64 32 16")
os.system("python3 pytorch.py 64 64 32 16")
os.system("rm -f in*");
os.system("rm -f weights*");

os.system("python3 datagen.py 64 64 64 32")
os.system("./coconv 64 64 64 32")
os.system("python3 pytorch.py 64 64 64 32")
os.system("rm -f in*");
os.system("rm -f weights*");

os.system("python3 datagen.py 64 64 100 100")
os.system("./coconv 64 64 100 100")
os.system("python3 pytorch.py 64 64 100 100")
os.system("rm -f in*");
os.system("rm -f weights*");

os.system("python3 datagen.py 100 100 100 100")
os.system("./coconv 100 100 100 100")
os.system("python3 pytorch.py 100 100 100 100")
os.system("rm -f in*");
os.system("rm -f weights*");

os.system("python3 datagen.py 128 128 100 100")
os.system("./coconv 128 128 100 100")
os.system("python3 pytorch.py 128 128 100 100")
os.system("rm -f in*");
os.system("rm -f weights*");

os.system("python3 datagen.py 256 256 100 100")
os.system("./coconv 256 256 100 100")
os.system("python3 pytorch.py 256 256 100 100")
os.system("rm -f in*");
os.system("rm -f weights*");

os.system("python3 datagen.py 500 500 100 100")
os.system("./coconv 500 500 100 100")
os.system("python3 pytorch.py 500 500 100 100")
os.system("rm -f in*");
os.system("rm -f weights*");
