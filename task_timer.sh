#!/bin/bash

PROGRESS_FILE="progress1.txt"

# Function to format time in HH:MM:SS
function format_time {
    local T=$1
    printf '%02d:%02d:%02d' $((T/3600)) $(( (T/60)%60)) $((T%60))
}

# Function to save progress
function save_progress {
    cat <<EOL > "$PROGRESS_FILE"
TOTAL_TASKS=$TOTAL_TASKS
TASKS_PER_SPRINT=$TASKS_PER_SPRINT
TASK_DURATION=$TASK_DURATION
USERNAME="$USERNAME"
CURRENT_TASK=$CURRENT_TASK
START_TIME=$START_TIME
TIME_SPENT=$TIME_SPENT
EOL
}

# Load progress if exists
if [ -f "$PROGRESS_FILE" ]; then
    source "$PROGRESS_FILE"
    RESUMING=true
else
    # Initialize variables
    TOTAL_TASKS=12
    TASKS_PER_SPRINT=4
    TASK_DURATION=30   # in minutes
    USERNAME=""
    CURRENT_TASK=1
    START_TIME=$(date +%s)
    TIME_SPENT=0
    RESUMING=false
fi

SCHEDULED_TOTAL_TIME=$((TOTAL_TASKS * TASK_DURATION * 60))  # in seconds

# Welcome message and user login
if [ "$RESUMING" = false ]; then
    echo "Welcome to the Task Timer!"
    read -p "Please enter your name: " USERNAME
    echo "Hello, $USERNAME! Let's get started."
else
    echo "Welcome back, $USERNAME! Resuming from where you left off."
fi

# Main loop for tasks
while [ $CURRENT_TASK -le $TOTAL_TASKS ]
do
    echo
    echo "-------------------------------"
    echo "Task $CURRENT_TASK of $TOTAL_TASKS"
    echo "Press Enter to start the task."
    read -p ""
    TASK_START=$(date +%s)
    
    echo "Task $CURRENT_TASK started. Work on your task..."
    echo "When you finish, press Enter to mark the task as complete."
    read -p ""
    TASK_END=$(date +%s)
    
    # Calculate time taken for the task
    TASK_TIME=$((TASK_END - TASK_START))
    TIME_SPENT=$((TIME_SPENT + TASK_TIME))
    TASKS_COMPLETED=$((CURRENT_TASK))
    TASKS_LEFT=$((TOTAL_TASKS - CURRENT_TASK))
    
    # Display task completion information
    echo "Task $CURRENT_TASK completed in $(format_time $TASK_TIME)."
    echo "Tasks remaining: $TASKS_LEFT"
    
    # Estimate hours left
    if [ $TASKS_COMPLETED -gt 0 ]; then
        AVG_TIME_PER_TASK=$((TIME_SPENT / TASKS_COMPLETED))
        ESTIMATED_TIME_LEFT=$((AVG_TIME_PER_TASK * TASKS_LEFT))
        echo "Estimated time left based on your current speed: $(format_time $ESTIMATED_TIME_LEFT)"
    fi
    
    # Increment task counter
    CURRENT_TASK=$((CURRENT_TASK + 1))
    
    # Save progress
    save_progress
    
    # Breaks and sprints logic
    if [ $(( (CURRENT_TASK - 1) % TASKS_PER_SPRINT )) -eq 0 ] && [ $CURRENT_TASK -le $TOTAL_TASKS ]; then
        if [ $((CURRENT_TASK - 1)) -eq 12 ]; then
            echo "Time for a 45-minute break. Press Enter to start the break timer."
            read -p ""
            BREAK_DURATION=$((45 * 60))
        else
            echo "Time for a 15-minute break. Press Enter to start the break timer."
            read -p ""
            BREAK_DURATION=$((15 * 60))
        fi
        
        # Save progress before break
        save_progress
        
        # Break timer with countdown
        BREAK_START=$(date +%s)
        BREAK_END=$((BREAK_START + BREAK_DURATION))
        echo "Break started at $(date -d @$BREAK_START '+%H:%M:%S'). It will end at $(date -d @$BREAK_END '+%H:%M:%S')."

        while [ $(date +%s) -lt $BREAK_END ]; do
            REMAINING=$((BREAK_END - $(date +%s)))
            echo -ne "Break time remaining: $(format_time $REMAINING)\r"
            sleep 1
        done
        echo -e "\nBreak over! Let's resume work."
    fi
done

# Total time taken
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo
echo "==============================="
echo "Great job, $USERNAME!"
echo "You completed all $TOTAL_TASKS tasks in $(format_time $TOTAL_TIME)."
echo "Have a wonderful rest of your day!"
echo "==============================="

# Remove progress file since tasks are completed
rm -f "$PROGRESS_FILE"
