#!/bin/bash
a=0
for i in *.wav; do
  new=$(printf "%d.wav" "$a")
  mv "$i" "$new"
  let a=a+1
done
