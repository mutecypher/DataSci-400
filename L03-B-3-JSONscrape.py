"""
# UW Data Science
# Please run code snippets one at a time to understand what is happening.
# Snippet blocks are sectioned off with a line of ####################
"""

# we can use a modified version of the base scraping format we learned earlier 
# to parse JSON format

# let's import the libraries we'll need

import urllib.request as ur # we'll give the urllib requests an easy to type alias 
import json
##########################

# now we'll setup a URL to pull down information from
# let's take a a look at this web site to see what we're working with 
url = "https://www.govtrack.us/data/congress/113/votes/2013/s11/data.json"

# now we can nest the urllib request call inside the json.load function to 
# pull data off the web and load it from json into a python dictionary
data = json.load(ur.urlopen(url))

print(data)

# for more information on how to pull out JSON elements and nested elements
# see the tutorial on JSON format
#########################

json_array = '{"cheese": ["gouda", "brie", "swiss"]}'

python_list = json.loads(json_array)

python_list = json.load(json_array)