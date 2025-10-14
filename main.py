import sys
import argparse

from converter import convert, init_logging


parser = argparse.ArgumentParser(description='Convert file to ome format')
parser.add_argument('--inputfile', required=True, help='input file')
parser.add_argument('--outputfolder', required=True, help='output folder')
parser.add_argument('--altoutputfolder', help='alternative output folder')
parser.add_argument('--outputformat', help='output format version', default='omezarr2')
parser.add_argument('--show_progress', action='store_true')
parser.add_argument('--verbose', action='store_true')
# Allow additional arguments for source-specific parameters (e.g., --plateid)
parser.add_argument('--plateid', help='Incucyte plate ID (optional)')
args = parser.parse_args()

init_logging('db_to_zarr.log', verbose=args.verbose)

# Build source-specific kwargs
source_kwargs = {}
if hasattr(args, 'plateid') and args.plateid:
    source_kwargs['plate_id'] = args.plateid

result = convert(
    args.inputfile,
    args.outputfolder,
    alt_output_folder=args.altoutputfolder,
    output_format=args.outputformat,
    show_progress=args.show_progress,
    verbose=args.verbose,
    **source_kwargs
)

if result and result != '{}':
    print(result)
    sys.exit(0)
else:
    print('Error')
    sys.exit(1)
