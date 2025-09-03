import sys
import argparse

from converter import convert, init_logging


parser = argparse.ArgumentParser(description='Convert file to ome format')
parser.add_argument('--inputfile', required=True, help='input file')
parser.add_argument('--outputfolder', required=True, help='output folder')
parser.add_argument("--inputformat", help="input file format")
parser.add_argument('--altoutputfolder', help='alternative output folder')
parser.add_argument('--outputformat', help='output format version', default='omezarr2')
parser.add_argument('--show_progress', action='store_true')
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()

init_logging('db_to_zarr.log', verbose=args.verbose)

result = convert(
    args.inputfile,
    args.outputfolder,
    input_format = args.inputformat,
    alt_output_folder = args.altoutputfolder,
    output_format = args.outputformat,
    show_progress = args.show_progress,
    verbose = args.verbose
)

if result and result != '{}':
    print(result)
    sys.exit(0)
else:
    print('Error')
    sys.exit(1)
