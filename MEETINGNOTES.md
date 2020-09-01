# Meeting Notes

This is a markdown file of for the Learning Consensus weekly syncs

## Tuesday August 25th, 2020 - Weekly Sync

Yash has updated simulation to continue until the honest agents commit - protocol running with 2 views. Byzantine agent delays on first view and on the second view with a new (honest) leader, the honest agents commit

TODO:

 - Run simulation without any penalties - we want to handhold as little as possible
 - Give a verify high reward for safety violation
    - Byzantine agent should be unable to exploit this attack if the byzantine agent is not the leader in consecutive rounds
 - Run simulation with 5 agents
 - Trenton will work on evolutionary search in this environment