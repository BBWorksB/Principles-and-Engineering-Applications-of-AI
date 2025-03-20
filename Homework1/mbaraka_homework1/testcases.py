from warehousebot import * 

def testcase1(): 
    # list of items 
    items = [Item(height = 5), Item(height = 4), Item(height = 3), Item(height = 2), Item(height = 6), Item(height = 7)] 
    # list of stacks 
    stacks = [Stack([], 10, name = "A"), Stack([], 15, name = "B"), Stack([], 20, name = "C")] 
    
    # totalcost = question1(items, stacks) # 18 to beat 
    # totalcost = question2(items, stacks) # 52 to beat 
    # totalcost = question3(items, stacks) # 65 to beat 
    stack = Stack([], 27, name = "D") 
    totalcost = question4(items, stack, items[0]) # 405 to beat 
    print(totalcost) 

def testcase2(): 
    # list of items 
    items = [Item(height = 6), Item(height = 7), Item(height = 1), Item(height = 3), Item(height = 5), Item(height = 2)] 
    # list of stacks 
    stacks = [Stack([], 10, name = "A"), Stack([], 15, name = "B"), Stack([], 20, name = "C"), Stack([], 21, name = "D")]  
    
    totalcost = question1(items, stacks) # 18 to beat 
    print(totalcost) 

def testcase3(): 
    # list of items 
    items = [Item(height = 6), Item(height = 6), Item(height = 8), Item(height = 7)] 
    # list of stacks 
    stacks = [Stack([], 15, name = "B"), Stack([], 20, name = "C")] 
    
    totalcost = question1(items, stacks) # 18 to beat 
    print(totalcost) 

if __name__ == "__main__": 
    testcase1() 
    testcase2() 
    testcase3() 
