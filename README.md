# Jack's Car Rental

Jack manages two locations for a nationwide car rental company. Each day, some number of customers arrive at each location to rent cars. If Jack has a car available, he rents it out and credited $10 by the national company. If he is out of cars at that location, then the business is lost. To help ensure that cars are available where they are needed, Jack can move them between the thwo locations overnight, at a cost of $2 per car moved.  To simplyfy the problem slightly, we assume that there can be no more than 20 cars at each location and a maximum of five cars can be moved from one location to the other in one night. Using policy iteration or value iteration, find best policy where Jack can make the most money.

### Define Problem
- States: Two locations, maximum of 20 cars at each

- Actions: Move up to 5 cars between locations overnight(cost of $2 per car moves)
    - +4: move 4 cars from 1st to 2nd\
    - -4: move 4 cars from 2nd to 1st
- Reward: $10 for each car rented
- Transitions: Cars returned and requested randomly
    - request and return follows Poisson distribution
    - 1st location: average requests = 3, averate returns = 3
    - 2nd  location: average requests = 4, average returns = 2

