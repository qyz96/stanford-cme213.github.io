#!/bin/bash

time_NoUser=0
# Time of no login after which to shut down
thre_NoUser=30

time_LowLoad=0
# Load threshold below which cpu is deemed idle
thre_LowLoad=0.3
# Time of idle after which to shut down
thre_TimeIdle=20

while [ 1 ]
do
    if [[ -z `who | grep -v tmux` ]]; then
        ((time_NoUser++))
    else
        time_NoUser=0
    fi
 
    if [[ $time_NoUser -ge $thre_NoUser ]]; then
        poweroff
    fi

    # average cpu load in the last 15 minutes
    load=$(uptime | sed -e 's/.*load average: //g' | awk '{ print $3 }')
    if (( $(echo "$thre_LowLoad $load" | awk '{print ($1 > $2)}') )); then
        ((time_LowLoad++))
    else
        time_LowLoad=0
    fi

    if [[ $time_LowLoad -ge $thre_TimeIdle ]]; then
        poweroff
    fi

    sleep 60
done
