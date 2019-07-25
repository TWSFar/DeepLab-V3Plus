import argparse
import sys

pass

def func():
    for i in range(0,3):
        yield i
 
f = func()
print(next(f))

print(next(f))

print(next(f))
