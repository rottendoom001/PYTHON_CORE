#!/bin/bash
CONTADOR=0
for FILE in *; do
  mv $FILE audio_$CONTADOR
  let CONTADOR=CONTADOR+1
done
