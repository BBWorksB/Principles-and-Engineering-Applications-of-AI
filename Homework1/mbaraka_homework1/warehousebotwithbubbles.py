from typing import List, Dict, Tuple 
from random import randint 

class Item: 
    def __init__(self, id: int = None, height: int = None) -> None: 
        if height == None: 
            raise ValueError("Height must be specified") 
        if id == None: 
            self.id = randint(10000, 100000) 
        else: 
            self.id = id 
        self.height = height # also the weight 
    
    def __repr__(self) -> str: 
        return f"Item({self.id}, {self.height})" 
    
    def __eq__(self, other) -> bool: 
        return self.id == other.id and self.height == other.height 

class Stack: 
    def __init__(self, items: List[Item], limit: int, name = None) -> None: 
        self.items = items 
        self.height = sum([item.height for item in items]) 
        self.limit = limit 
        self.name = name 
    
    def __repr__(self) -> str: 
        return f"Stack({self.name}-{self.items})" 
    
    def __eq__(self, other) -> bool: 
        return self.items == other.items 
    
    def add_item(self, item: Item, question: str) -> None: 
        if self.height + item.height > self.limit: 
            raise ValueError("Item too tall to add to stack") 
        
        # compute the cost 
        if question == "1": 
            cost = self.height 
        elif question == "2" or question == "3" or question == "4": 
            cost = self.height * item.height # height prior to adding times weight 
        
        # place the item on the stack 
        self.items.append(item) 
        self.height += item.height 
        
        return cost 
    
    def retrieve_item(self, item: Item) -> None: # only needed for question 4 
        if item not in self.items: 
            raise ValueError("Item not in stack") 
        
        # compute the cost 
        index = self.items.index(item) 
        totalweight = 0 
        for i in range(index, len(self.items)): 
            totalweight += self.items[i].height 
        totalheightsummed = 0 
        for i in range(index): # not including the item 
            totalheightsummed += self.items[i].height 
        cost = totalweight * totalheightsummed 
        additionalcost = (totalweight - item.height) * totalheightsummed # the cost of putting the items above back 
        cost += additionalcost 
        # print(cost) 
        
        return cost 

def question1(items: List[Item], stacks: List[Stack]) -> int: 
    totalcost = 0 
    
    # add your code here 
    for item in items:
        # Sort stacks by current height (ascending) to use the lowest available first
        stacks.sort(key=lambda s: s.height)
        
        for stack in stacks:
            if stack.height + item.height <= stack.limit:
                totalcost += stack.add_item(item, "1")
                break  
    return totalcost 



def question2(items: List[Item], stacks: List[Stack]) -> int: 
    totalcost = 0 
    
    # add your code here 
    #  Sort items by height in descending order to place larger items first
    sorted_items = sorted(items, key=lambda x: x.height, reverse=True)
    
    for item in sorted_items:
        # find valid stacks that won't exceed height limit
        valid_stacks = [s for s in stacks if s.height + item.height <= s.limit]
        if not valid_stacks:
            continue
            
        # Calculate cost for each valid stack
        min_cost = float('inf')
        best_stack = None
        for stack in valid_stacks:
            # Consider both current height and future impact
            cost = stack.height * item.height
            if cost < min_cost:
                min_cost = cost
                best_stack = stack
                
        if best_stack:
            totalcost += best_stack.add_item(item, "2")
            
    return totalcost


def question3(items: List[Item], stacks: List[Stack]) -> int: 
    totalcost = 0 
    
    # add your code here 
    for item in items:
        min_cost = float('inf')
        selected_stack = None
        for stack in stacks:
            if stack.height + item.height <= stack.limit:
                cost = stack.height * item.height
                if cost < min_cost:
                    min_cost = cost
                    selected_stack = stack
        if selected_stack:
            totalcost += selected_stack.add_item(item, "3")
    return totalcost 

def question4(items: List[Item], stack: Stack, item_to_pullout: Item) -> int:
    assert item_to_pullout in items, "Item to pull out must be in the list of items"
    totalcost = 0
    # Separate the item to be removed and sort the others
    other_items = [item for item in items if item != item_to_pullout]
    sorted_other = sorted(other_items, key=lambda x: x.height)
    sorted_items = sorted_other + [item_to_pullout]
    # Place all items
    for item in sorted_items:
        totalcost += stack.add_item(item, "4")
    # Calculate removal cost
    remove_cost = stack.retrieve_item(item_to_pullout)
    totalcost += remove_cost
    return totalcost


if __name__ == "__main__": 
    # list of items 
    items = [Item(height = 5), Item(height = 4), Item(height = 3), Item(height = 2), Item(height = 6), Item(height = 7)] 
    # list of stacks 
    stacks = [Stack([], 10, name = "A"), Stack([], 15, name = "B"), Stack([], 20, name = "C")] 
    
    totalcost = question1(items, stacks) # 18 to beat 
    # totalcost = question2(items, stacks) # 52 to beat 
    # totalcost = question3(items, stacks) # 65 to beat 
    stack = Stack([], 27, name = "D") 
    # totalcost = question4(items, stack, items[0]) # 405 to beat 
    print(totalcost) 
