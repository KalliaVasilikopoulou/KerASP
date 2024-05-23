f = open("user_scripts/grocery_price_list.txt", "r")
lines = f.readlines()[2:]   #the first lines correpond to the title of the grocery list

prices = {}

for line in lines: 
    line = line.replace(' €/kg','').replace(':','').replace('.','')
    line = line.split(' ')
    prices[line[1]] = int(line[2])

f.close()
