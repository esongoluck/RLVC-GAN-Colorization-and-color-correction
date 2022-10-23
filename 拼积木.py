import copy

m=0
bn=5

gc=10
gr=10
blocks=[]

vis=[]
def clearscreen(numlines=100):
    """Clear the console.
    numlines is an optional argument used only as a fall-back.
    """
    import os
    if os.name == "posix":
        # Unix/Linux/MacOS/BSD/etc
        os.system('clear')
    elif os.name in ("nt", "dos", "ce"):
        # DOS/Windows
        os.system('CLS')
    else:
        # Fallback for other operating systems.
        print ('\n' * numlines)
import sys, time

def delete_last_line():
    sys.stdout.write('\x1b[1A')
    sys.stdout.write('\x1b[2K')
def showG():
    for ri in range(gr):
        for ci in range(gc):
            if G[ri][ci]>0:
                print(chr(ord('a')+G[ri][ci]-1),end="")#chr(ord('a')+G[ri][ci]-1)
            else :
                print("'",end="")
        print("")
    print("")


def b_c_turn(block_in):
    block=copy.deepcopy(block_in)
    for ri in range(bn):
        for ci in range(int(bn/2)):
            t=block[ri][ci]
            block[ri][ci]=block[ri][bn - ci - 1]
            block[ri][bn - ci - 1]=t
    return block

def b_e_turn(block_in):
    block=copy.deepcopy(block_in)
    for ri in range(bn):
        for ci in range(ri,bn):
            t=block[ri][ci]
            block[ri][ci]=block[ci][ri]
            block[ci][ri]=t
    return block

def b_turn(block_in):
    block=copy.deepcopy(block_in)
    block=b_e_turn(block)
    block=b_c_turn(block)
    return block

def make_truns_b():
    for bi in range(m):
        for dir in range(1,4):
            blocks[bi][dir]=b_turn(blocks[bi][dir-1])
        blocks[bi][4] = b_c_turn(blocks[bi][4 - 1])
        for dir in range(5,8):
            blocks[bi][dir]=b_turn(blocks[bi][dir-1])
def init_b():

    for bi in range(m):
        for dir in range(8):
            r0=100
            c0=100
            block = [[0 for _ in range(bn)] for _ in range(bn)]
            for ri in range(bn):
                for ci in range(bn):
                    if blocks[bi][dir][ri][ci]>0:
                        r0=min(r0,ri)
                        c0=min(c0,ci)
            for ri in range(bn-r0):
                for ci in range(bn-c0):
                    block[ri][ci]=blocks[bi][dir][ri+r0][ci+c0]
            blocks[bi][dir]=block
def up(nowi,bi,dir):
    if vis[bi]==0:
        return False
    c_d=0
    nowr=int(nowi/gc)
    nowc=int(nowi%gc)
    for ci in range(bn-1,0-1,-1):
        if blocks[bi][dir][0][ci]>0:
            c_d=ci
    nowc-=c_d
    for ri in range(bn):
        for ci in range(bn):
            if blocks[bi][dir][ri][ci]>0 and (ri+nowr>=gr or ci+nowc>=gc or ri+nowr<0 or ci+nowc<0 or G[ri+nowr][ci+nowc]>0):
                return False

    for ri in range(bn):
        for ci in range(bn):
            if blocks[bi][dir][ri][ci]>0:
                G[ri+nowr][ci+nowc]=bi+1
    vis[bi]-=1
    return True

def down(nowi,bi,dir):
    c_d=0
    nowr=int(nowi/gc)
    nowc=int(nowi%gc)
    for ci in range(bn-1,0-1,-1):
        if blocks[bi][dir][0][ci]>0:
            c_d=ci
    nowc-=c_d

    for ri in range(bn):
        for ci in range(bn):
            if blocks[bi][dir][ri][ci]>0:
                G[ri+nowr][ci+nowc]=0
    vis[bi]+=1

def find_next(i):
    i+=1
    if i==gr*gc :
        return i
    nowr=int(i/gc)
    nowc=int(i%gc)
    #print("rc",nowr,nowc)
    if G[nowr][nowc]>0:
        return find_next(i)
    return i

def pin(nowi):

    if nowi==gr*gc and find_next(0)==gr*gc:
        return 1
    for bi in range(m):
        for dir in range(8):
            if up(nowi,bi,dir):
                if pin(find_next(nowi)):
                    return 1
                down(nowi,bi,dir)
    #showG()
    #print( nowi)
    showG()
    return 0



if __name__ == '__main__':
    fc = open("test.txt", 'r')
    contents=fc.read()
    fc.close()
    fc=contents.split()
    fn=0
    m=int(fc[fn])
    fn+=1
    blocks=[[[[0 for _ in range(bn)] for _ in range(bn)] for _ in range(8)] for _ in range(m)]
    G=[[0 for _ in range(gc)] for _ in range(gr)]
    vis=[1 for _ in range(m)]
    vis[12]=4

    for r in range(gr):
        for c in range(gc):
            G[r][c]=int(fc[fn])*20
            fn+=1

    showG()
    for i in range(m):
        for r in range(bn):
            for c in range(bn):
                blocks[i][0][r][c]=int(fc[fn])*(i+1)
                fn+=1

    make_truns_b()
    init_b()

    print("start!!!")
    print(pin(find_next(-1)))
    showG()