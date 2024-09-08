#!/bin/bash

USER="k2257777"
SUBMIT_TIME=$(date +%s)

# ANSI color codes
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to display time in a human-readable format
format_time() {
    local seconds=$1
    local days=$((seconds / 86400))
    local hours=$(( (seconds % 86400) / 3600 ))
    local minutes=$(( (seconds % 3600) / 60 ))
    local remaining_seconds=$((seconds % 60))
    
    if [ $days -gt 0 ]; then
        echo "${days}d ${hours}h ${minutes}m ${remaining_seconds}s"
    elif [ $hours -gt 0 ]; then
        echo "${hours}h ${minutes}m ${remaining_seconds}s"
    elif [ $minutes -gt 0 ]; then
        echo "${minutes}m ${remaining_seconds}s"
    else
        echo "${remaining_seconds}s"
    fi
}

# Function to get basic job info
get_job_info() {
    squeue -h -u $USER -o "%i %T %M %l %r"
}

# Function to get detailed job info
get_detailed_job_info() {
    local jobid=$1
    scontrol show job $jobid
}

# Function to kill a job
kill_job() {
    local jobid=$1
    scancel $jobid
    echo -e "${YELLOW}Job $jobid has been terminated.${NC}"
}

# Function to draw a box
draw_box() {
    local width=$1
    local title=$2
    printf "┌─%s─┐\n" "$(printf '%.0s─' $(seq 1 $((width-4))))"
    printf "│ %-*s │\n" $((width-4)) "$title"
    printf "├─%s─┤\n" "$(printf '%.0s─' $(seq 1 $((width-4))))"
}

# Function to close a box
close_box() {
    local width=$1
    printf "└─%s─┘\n" "$(printf '%.0s─' $(seq 1 $((width-4))))"
}

# Initialize job status tracking
declare -A job_start_times

while true; do
    current_time=$(date +%s)
    wait_time=$(format_time $((current_time - SUBMIT_TIME)))
    
    clear  # Clear the screen for better readability
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║                   Slurm Job Monitor for $USER                 ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo -e "${GREEN}Current monitoring time: $wait_time${NC}"
    echo ""
    
    job_info=$(get_job_info)
    
    if [ -z "$job_info" ]; then
        echo -e "${YELLOW}No jobs found for user $USER${NC}"
    else
        while IFS= read -r line; do
            read -r jobid state runtime timelimit reason <<< "$line"
            
            if [ "$state" = "RUNNING" ] && [ -z "${job_start_times[$jobid]}" ]; then
                job_start_times[$jobid]=$current_time
            fi
            
            draw_box 70 "Job ID: $jobid - State: $state"
            printf "│ %-66s │\n" "Runtime: $runtime"
            printf "│ %-66s │\n" "Time Limit: $timelimit"
            
            if [ -n "${job_start_times[$jobid]}" ]; then
                job_wait_time=$(format_time $((job_start_times[$jobid] - SUBMIT_TIME)))
                printf "│ %-66s │\n" "Started after waiting: $job_wait_time"
            else
                printf "│ %-66s │\n" "Reason: $reason"
            fi
            
            # Get and display detailed job info
            detailed_info=$(get_detailed_job_info $jobid)
            
            # Extract and display the command
            command=$(echo "$detailed_info" | grep "Command=" | cut -d'=' -f2-)
            printf "│ %-66s │\n" "Command: ${command:0:60}..."
            
            # Extract and display the working directory
            workdir=$(echo "$detailed_info" | grep "WorkDir=" | cut -d'=' -f2-)
            printf "│ %-66s │\n" "Working Directory: ${workdir:0:50}..."
            
            # Extract and display the number of nodes and CPUs
            nodes=$(echo "$detailed_info" | grep "NumNodes=" | cut -d'=' -f2- | cut -d' ' -f1)
            cpus=$(echo "$detailed_info" | grep "NumCPUs=" | cut -d'=' -f2- | cut -d' ' -f1)
            printf "│ %-66s │\n" "Nodes: $nodes, CPUs: $cpus"
            
            # Extract and display GPU information if available
            gpu_info=$(echo "$detailed_info" | grep "GRES=" | cut -d'=' -f2-)
            if [ -n "$gpu_info" ]; then
                printf "│ %-66s │\n" "GPU Info: $gpu_info"
            fi
            
            close_box 70
            echo ""
        done <<< "$job_info"
    fi
    
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║                            Options                             ║${NC}"
    echo -e "${BLUE}╠════════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${BLUE}║${NC} - Press ${GREEN}Enter${NC} to refresh                                        ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC} - Type ${RED}'q'${NC} to quit                                             ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC} - Type a ${YELLOW}job ID${NC} to terminate that job                           ${BLUE}║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo -n "Enter your choice: "
    
    read -t 60 choice
    
    case $choice in
        q|Q)
            echo -e "${RED}Exiting...${NC}"
            exit 0
            ;;
        "")
            # User pressed Enter, just continue to next iteration
            ;;
        *)
            if [[ $choice =~ ^[0-9]+$ ]]; then
                kill_job $choice
                sleep 2  # Give some time for the job to be terminated before refreshing
            else
                echo -e "${RED}Invalid input. Please try again.${NC}"
                sleep 2
            fi
            ;;
    esac
done