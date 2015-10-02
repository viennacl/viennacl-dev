#!/bin/bash

FAST="fast"

if [[ $1 = $FAST || $2 = $FAST || $3 = $FAST ]]; then
    FAST="fast"
else
    FAST=""
fi

printf %s "$(echo flash is $FAST)"
