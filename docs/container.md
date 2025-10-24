# General container requirements

## Input arguments
* --inputfile (required): Path to the input file
* --outputfolder (required): Output directory for converted files
* --altoutputfolder (optional): Alternative output directory
* --show_progress (optional): Show progress bar during conversion (boolean value)

## Output format
The container should return the following output as JSON string

A list containing a dictionary of:
* name: base name of the created or relevant file (without extension)
* full_path: absolute path to the output file
* alt_path: absolute path to the file in altoutputfolder (if used and file exists), else null
* keyvalues (optional): a list with a dictionary containing per-channel intensity stats and optional metadata


### keyvalues example
```
"keyvalues":
[
  {
    "channel_mins": [1097, 2257, 335, 175],
    "channel_maxs": [7423, 5907, 10261, 3716],
    "channel_display_black_values": [1179, 2357, 372, 202],
    "channel_display_white_values": [6798, 5641, 7002, 2969]
  }
]
```
