# Mercury

Happy path:
User inputs in natural language the bus they want to know about, or the stop they want to know about, or the direction they want to know about. The user has to give key information such as the:

Key Information (OR)
1. Bus Line
2. Stop Name
3. Direction AND Current Location
4. Time


VERSION 1.0
Pull the MTA bus time

*solution*:
simple pull using request and API key and display the data

*success*: 
the data is pulled and displayed

VERSION 1.1
*problem*: the data is not readable
*commit*:Parse the data into a readable format

*solution*:
use response.json() to get the data and then use the data to display the information we want.

*success*: 
the data is pulled and displayed in a readable format

VERSION 1.2
*userstory*
user wants to know the next bus at a certain stop in a certain direction.
*problem*:
the data is currently restricted to the bus line M57 and displays all the buses on that line. we need to add the ability to search for a certain stop and direction.
*solution*:
we need to query the data for the stop and direction that the user wants to know about
*test*:
1. allow filtering by stop name and directoin
2. return structured data for each matching bus
3. include test cases

VERSION 1.2.1
*userstory*
users want to know the bus in certain direction at a certain stop

*solution*:
Added parse_direction function to handle various direction inputs
Accepts natural language direction inputs (rule-based), you can add more rules if you want or use a LLM to parse the direction

Added more test cases with different direction formats
The code now accepts inputs like:
"east", "towards 1st ave", "e", "eb"
"west", "towards broadway", "w", "wb"

VERSION 1.2.2 (open it up)
*userstory*
users want to know the next bus at a any stop in any direction

*solution*:
light weigth LLM to handle the natural language query and the pass to get the next bus function

