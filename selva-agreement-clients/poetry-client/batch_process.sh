#!/bin/bash
# var for session name (to avoid repeated occurences)
PYTHONBIN=/home/berstearns/.cache/pypoetry/virtualenvs/poetry-client-UVpoFPKR-py3.12/bin/python3
sn=xyz

# Start the session and window 0 in /etc
#   This will also be the default cwd for new windows created
#   via a binding unless overridden with default-path.
#tmux new-session -s "$sn" -n tokenization -d

# Expect to be in the selva-agreements/clients/poetry-clinet folder 
# Create a bunch of windows, one for each data split
DATASPLITS=()
INPUT_BATCH_FOLDER="./outputs/EFCAMDAT/training_splits_efcamdat/test"
OUTPUT_BATCH_FOLDER="./outputs/EFCAMDAT/tokenization_batch/test/"
for FILENAME in `ls $INPUT_BATCH_FOLDER`;
do
	EXPECTED_OUTPUT='${OUTPUT_BATCH_FOLDER}/${FILENAME}'
	TEST=`wc -l $EXPECTED_OUTPUT 2> /dev/null | awk -F ' ' '{ print $1 }' `  
	if [ -n "$TEST" ] && [ "$TEST" -gt 0 ] 
	then
		LINECOUNT=`wc -l $EXPECTED_OUTPUT | awk -F ' ' '{ print $1 }'` 
	else
		echo "$FILENAME will be processed";
		DATASPLITS+=( $FILENAME )
	fi
done
for i in ${!DATASPLITS[@]}; 
do
    FILENAME=${DATASPLITS[$i]}
    FILEPATH="${INPUT_BATCH_FOLDER}/${FILENAME}"
    COMMAND="${PYTHONBIN} sent_tokenize.py ${FILEPATH}" 
    echo $i "->" ${DATASPLITS[$i]}
    #tmux new-window -t "$sn:$((i+1))" -n "${FILENAME:(-3)}" "zsh -c script.py"
    $COMMAND &
done

# Set the default cwd for new windows (optional, otherwise defaults to session cwd)
#tmux set-option default-path /

# Select window #1 and attach to the session
#tmux select-window -t "$sn:0"
#tmux -2 attach-session -t "$sn"
