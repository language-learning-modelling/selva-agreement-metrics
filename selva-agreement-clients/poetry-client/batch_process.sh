#!/bin/bash
# var for session name (to avoid repeated occurences)
PYTHONBIN=/home/berstearns/.cache/pypoetry/virtualenvs/poetry-client-UVpoFPKR-py3.12/bin/python3
SCRIPTFP=sent_tokenize.py
MAX_NUM_TO_PROCESS=10
sn=xyz

# Start the session and window 0 in /etc
#   This will also be the default cwd for new windows created
#   via a binding unless overridden with default-path.
#tmux new-session -s "$sn" -n tokenization -d

# Expect to be in the selva-agreements/clients/poetry-clinet folder
# Create a bunch of windows, one for each data split
DATASPLITS=()

####################
## CELVA FULL     ##
####################
#SPLIT=""
#DATASET="CELVA"
#INPUT_BATCH_FOLDER="./outputs/${DATASET}/splits/"
#OUTPUT_BATCH_FOLDER="./outputs/${DATASET}/tokenization_batch/"

####################
## EFCAMDAT TRAIN ##
####################
SPLIT="train"
DATASET="EFCAMDAT"
INPUT_BATCH_FOLDER="./outputs/${DATASET}/splits/${SPLIT}_json"
OUTPUT_BATCH_FOLDER="./outputs/${DATASET}/tokenization_batch/${SPLIT}"

####################
## EFCAMDAT TEST  ##
####################
# SPLIT="test"
# DATASET="EFCAMDAT"
# INPUT_BATCH_FOLDER="./outputs/${DATASET}/splits/${SPLIT}_json"
# OUTPUT_BATCH_FOLDER="./outputs/${DATASET}/tokenization_batch/${SPLIT}"
for FILENAME in $(ls $INPUT_BATCH_FOLDER -p | grep -v /); do
  EXPECTED_OUTPUT=${OUTPUT_BATCH_FOLDER}/${FILENAME}
  TEST=$(wc -l $EXPECTED_OUTPUT 2>/dev/null | awk -F ' ' '{ print $1 }')
  if [ -n "$TEST" ] && [ "$TEST" -gt 0 ]; then
    LINECOUNT=$(wc -l $EXPECTED_OUTPUT | awk -F ' ' '{ print $1 }')
  else
    if [ "${#DATASPLITS[@]}" -lt $MAX_NUM_TO_PROCESS ]; then
      echo "$FILENAME will be processed"
      DATASPLITS+=($FILENAME)
    fi
  fi
done
for i in ${!DATASPLITS[@]}; do
  FILENAME=${DATASPLITS[$i]}
  FILEPATH="${INPUT_BATCH_FOLDER}/${FILENAME}"
  CONFIG=$(
    jo -p input_fp=$FILEPATH ud_model_fp='./udpipe_models/english-ewt-ud-2.5-191206.udpipe' \
      output_folder=$OUTPUT_BATCH_FOLDER text_column='text'
  )
  COMMAND="${PYTHONBIN} -W ignore ${SCRIPTFP} $CONFIG"
  # -i
  echo $i "->" ${DATASPLITS[$i]}
  echo $CONFIG
  # echo $CONFIG
  #tmux new-window -t "$sn:$((i+1))" -n "${FILENAME:(-3)}" "zsh -c script.py"
  $COMMAND &
done

# Set the default cwd for new windows (optional, otherwise defaults to session cwd)
#tmux set-option default-path /

# Select window #1 and attach to the session
#tmux select-window -t "$sn:0"
#tmux -2 attach-session -t "$sn"
