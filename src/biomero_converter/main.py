import sys
import argparse
import multiprocessing

from biomero_converter.converter import convert, init_logging
from biomero_converter.metadata_view import VIEWS, format_metadata, read_metadata
from biomero_converter.parameters import DEFAULT_OUTPUT_FORMAT


def main():
    multiprocessing.freeze_support()   # worker processes (e.g. iSyntax reads) in a frozen (PyInstaller) app
    parser = argparse.ArgumentParser(description='Convert file to ome format')
    parser.add_argument('--inputfile', required=True, help='input file')
    parser.add_argument('--outputfolder', help='output folder (required to convert)')
    parser.add_argument('--metadata', nargs='?', const='all', choices=('all',) + VIEWS,
                        help='only print the metadata of the input, as YAML, without converting: the source '
                             '(vendor) metadata, the model metadata it maps to, the spatial metadata (pixel size, '
                             'position, rotation) the conversion writes, or all (default)')
    parser.add_argument('--altoutputfolder', help='alternative output folder')
    parser.add_argument('--outputformat', default=DEFAULT_OUTPUT_FORMAT,
                        help=f'output format: omezarr04, omezarr05, omezarr06 or ometiff (default: {DEFAULT_OUTPUT_FORMAT})')
    parser.add_argument('--show_progress', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    # Allow additional arguments for source-specific parameters (e.g., --plateid)
    parser.add_argument('--plateid', help='Incucyte plate ID (optional)')
    parser.add_argument('--image_uuid', help='Leica image UUID (optional, default: all images) '
                                               'or DICOM SeriesInstanceUID (optional, default: the first series)')
    parser.add_argument('--position', type=int, help='Leica position index, for a tile scan with gaps between its tiles')
    parser.add_argument('--crop_to_scan', action=argparse.BooleanOptionalAction, default=True,
                        help='MIRAX: crop to the scanned region, leaving out the unscanned (background) area '
                             '(default; --no-crop_to_scan: the whole slide)')
    args = parser.parse_args()
    if args.metadata is None and args.outputfolder is None:
        parser.error('--outputfolder is required to convert (or use --metadata to only print the metadata)')

    init_logging('db_to_zarr.log', verbose=args.verbose)

    # Build source-specific kwargs
    source_kwargs = {}
    if hasattr(args, 'plateid') and args.plateid:
        source_kwargs['plate_id'] = args.plateid
    if args.image_uuid:
        source_kwargs['image_uuid'] = args.image_uuid
    if args.position is not None:
        source_kwargs['position'] = args.position
    source_kwargs['crop_to_scan'] = args.crop_to_scan

    if args.metadata is not None:
        views = VIEWS if args.metadata == 'all' else (args.metadata,)
        metadata = read_metadata(args.inputfile, views, **source_kwargs)
        print(format_metadata(metadata if args.metadata == 'all' else metadata[args.metadata]))
        sys.exit(0)

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


if __name__ == '__main__':
    main()
