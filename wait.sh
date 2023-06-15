#!/bin/bash
# The PID to wait for is the first command-line argument
pid=$1

# Check if pid was provided
if [ -z "$pid" ]; then
    echo "Please provide a PID as an argument."
    exit 1
fi

# Wait for the process to end
while kill -0 $pid 2> /dev/null; do
  sleep 1
done

echo "Process $pid has ended. Waiting 2 minutes before shutting down."
sleep 120

# Shutdown the system
sudo shutdown -h now
