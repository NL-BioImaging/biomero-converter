import numpy as np
import re
from pathlib import Path
import tifffile
from datetime import datetime

from src.ImageSource import ImageSource
from src.util import strip_leading_zeros
from src.TiffSource import TiffSource


class IncucyteSource(ImageSource):
    """
    ImageSource implementation for Incucyte data

    Handles the specific directory structure:
    EssenFiles/ScanData/YYMM/DD/HHMM/XXXX/*.tif

    Filenames follow pattern: WELL-FIELD-CHANNEL.tif
    e.g., A1-1-C1.tif, B2-1-Ph.tif
    """

    def __init__(self, uri, metadata={}):
        super().__init__(uri, metadata)
        self.base_path = Path(self.uri)
        self.scan_data_path = self.base_path / "EssenFiles" / "ScanData"
        self._file_cache = {}
        # Default to True for filling missing images
        self.fill_missing_images = True

    def init_metadata(self):
        """Initialize all metadata from Incucyte structure"""
        self._get_experiment_metadata()
        self._scan_timepoints()
        self._get_well_info()
        self._get_channel_info()
        self._get_image_info()

        # Initialize properties like TiffSource does
        self.name = self.metadata.get("Name", "Incucyte_Experiment")
        self.dim_order = self.metadata.get("dim_order", "tczyx")
        self.dtype = self.metadata.get("dtype", np.uint16)
        self.pixel_size = self._get_pixel_size_dict()
        self.channels = self._format_channels_for_interface()
        self.is_plate = len(self.metadata.get("wells", {})) > 0
        self.wells = list(self.metadata.get("wells", {}).keys())
        self.rows = self.metadata.get("well_info", {}).get("rows", [])
        self.columns = self.metadata.get("well_info", {}).get("columns", [])

        return self.metadata

    def _get_experiment_metadata(self):
        """Extract experiment metadata from folder structure"""
        experiment_name = self.base_path.name
        self.metadata.update(
            {
                "Name": experiment_name,
                "Creator": "Incucyte",
                "DateCreated": datetime.now(),
                "dim_order": "tczyx",
            }
        )

    def _scan_timepoints(self):
        """Scan the Incucyte directory structure for timepoints"""
        timepoints = []
        wells = set()
        fields = set()
        channels = set()

        print(f"Scanning directory: {self.scan_data_path}")

        if not self.scan_data_path.exists():
            raise ValueError(f"Scan data path not found: {self.scan_data_path}")

        # Navigate through year/month directories (YYMM)
        for year_month in self.scan_data_path.iterdir():
            if not year_month.is_dir():
                continue
            # Navigate through day directories (DD)
            for day in year_month.iterdir():
                if not day.is_dir():
                    continue
                # Navigate through time directories (HHMM)
                for time_dir in day.iterdir():
                    if not time_dir.is_dir():
                        continue
                    # Navigate through fixed ID directories (XXXX)
                    for fixed_id in time_dir.iterdir():
                        if not fixed_id.is_dir():
                            continue

                        timepoint_path = fixed_id
                        timestamp = f"{year_month.name}_{day.name}_{time_dir.name}"

                        # Parse timestamp to datetime
                        try:
                            dt = datetime.strptime(timestamp, "%y%m_%d_%H%M")
                            if dt.year < 2000:
                                dt = dt.replace(year=dt.year + 2000)
                        except ValueError:
                            dt = None

                        timepoint_info = {
                            "path": timepoint_path,
                            "timestamp": timestamp,
                            "datetime": dt,
                            "index": len(timepoints),
                        }
                        timepoints.append(timepoint_info)

                        # Scan TIFF files in this timepoint
                        for tiff_file in timepoint_path.glob("*.tif"):
                            well, field, channel = self._parse_filename(tiff_file.name)
                            if well and field is not None and channel:
                                wells.add(well)
                                fields.add(field)
                                channels.add(channel)

        # Sort timepoints by datetime if available, otherwise by timestamp
        timepoints.sort(
            key=lambda x: x["datetime"] if x["datetime"] else x["timestamp"]
        )

        # Update indices after sorting
        for i, tp in enumerate(timepoints):
            tp["index"] = i

        self.metadata.update(
            {
                "timepoints": timepoints,
                "time_points": [tp["index"] for tp in timepoints],
                "wells_raw": sorted(wells),
                "fields_raw": sorted(fields),
                "channels_raw": sorted(channels),
            }
        )

        print(
            f"Found: {len(timepoints)} timepoints, {len(wells)} wells, {len(fields)} fields, {len(channels)} channels"
        )

    def _parse_filename(self, filename):
        """
        Parse Incucyte filename format: WELL-FIELD-CHANNEL.tif
        Examples: A1-1-C1.tif, B2-1-Ph.tif
        Returns: (well, field, channel)
        """
        pattern = r"([A-Z]\d+)-(\d+)-(.+)\.tif"
        match = re.match(pattern, filename)
        if match:
            well = match.group(1)
            field = int(match.group(2)) - 1  # Convert to 0-based indexing
            channel = match.group(3)
            return well, field, channel
        return None, None, None

    def _get_well_info(self):
        """Process well information and determine plate layout"""
        wells_raw = self.metadata["wells_raw"]

        if not wells_raw:
            raise ValueError("No wells found in data")

        # Parse well positions
        rows = set()
        cols = set()
        wells_dict = {}

        for well_name in wells_raw:
            row_letter = well_name[0]
            col_number = int(well_name[1:])

            rows.add(row_letter)
            cols.add(col_number)

            wells_dict[well_name] = {
                "Name": well_name,
                "row": ord(row_letter) - ord("A"),
                "column": col_number - 1,
                "ZoneIndex": len(wells_dict),
            }

        rows = sorted(rows)
        cols = sorted(cols)

        # Get image dimensions from first available image
        sample_image_info = self._get_sample_image_info()

        well_info = {
            "rows": rows,
            "columns": [str(c) for c in cols],
            "SensorSizeXPixels": sample_image_info["width"],
            "SensorSizeYPixels": sample_image_info["height"],
            "SitesX": 1,
            "SitesY": 1,
            "num_sites": len(self.metadata["fields_raw"]),
            "fields": [str(f) for f in self.metadata["fields_raw"]],
            "PixelSizeUm": sample_image_info["pixel_x"],
            "SensorBitness": sample_image_info["bits"],
            "max_sizex_um": sample_image_info["width"] * sample_image_info["pixel_x"],
            "max_sizey_um": sample_image_info["height"] * sample_image_info["pixel_y"],
        }

        self.metadata.update({"wells": wells_dict, "well_info": well_info})

    def _get_sample_image_info(self):
        """Get image dimensions and bit depth from first available TIFF"""
        for timepoint in self.metadata["timepoints"]:
            for tiff_file in timepoint["path"].glob("*.tif"):
                try:
                    # Use TiffSource to extract metadata
                    temp_tiff_source = TiffSource(str(tiff_file))
                    temp_tiff_source.init_metadata()

                    # Get pixel size from TiffSource
                    pixel_size = temp_tiff_source.get_pixel_size_um()

                    # Get actual image dimensions from the file
                    with tifffile.TiffFile(str(tiff_file)) as tif:
                        page = tif.pages[0]
                        array = page.asarray()

                    temp_tiff_source.close()

                    return {
                        "width": array.shape[1],
                        "height": array.shape[0],
                        "bits": array.dtype.itemsize * 8,
                        "dtype": array.dtype,
                        "pixel_x": pixel_size.get("x", 1.0),
                        "pixel_y": pixel_size.get("y", 1.0),
                    }
                except Exception as e:
                    print(f"Could not read sample image {tiff_file}: {e}")
                    continue

        # If no valid TIFF files found
        raise ValueError(
            f"No valid TIFF files found in experiment directory: {self.scan_data_path}"
        )

    def _get_channel_info(self):
        """Process channel information"""
        channels_raw = self.metadata["channels_raw"]
        channels = []

        channel_mapping = {
            "C1": {"label": "Green", "color": "00FF00"},
            "C2": {"label": "Red", "color": "FF0000"},
            "Ph": {"label": "Phase_Contrast", "color": "FFFFFF"},
            "P": {"label": "Phase_Contrast", "color": "FFFFFF"},
        }

        for i, channel_code in enumerate(channels_raw):
            channel_info = channel_mapping.get(
                channel_code, {"label": channel_code, "color": "FFFFFF"}
            )

            channels.append(
                {
                    "ChannelNumber": i,
                    "Dye": channel_info["label"],
                    "Color": f"#{channel_info['color']}",
                    "Emission": None,
                    "Excitation": None,
                    "code": channel_code,
                }
            )

        self.metadata.update({"channels": channels, "num_channels": len(channels)})

    def _get_image_info(self):
        """Get image-related metadata"""
        sample_info = self._get_sample_image_info()

        well_info = self.metadata["well_info"]
        max_data_size = (
            well_info["SensorSizeXPixels"]
            * well_info["SensorSizeYPixels"]
            * len(self.metadata["wells"])
            * well_info["num_sites"]
            * self.metadata["num_channels"]
            * len(self.metadata["time_points"])
            * (sample_info["bits"] // 8)
        )

        self.metadata.update(
            {
                "bits_per_pixel": sample_info["bits"],
                "dtype": sample_info["dtype"],
                "max_data_size": max_data_size,
            }
        )

    def _get_pixel_size_dict(self):
        """Get pixel size in TiffSource format"""
        well_info = self.metadata.get("well_info", {})
        pixel_size = well_info.get("PixelSizeUm", 1.0)
        return {"x": pixel_size, "y": pixel_size}

    def _format_channels_for_interface(self):
        """Format channels for interface compatibility"""
        channels = self.metadata.get("channels", [])
        return [
            {"label": ch["Dye"], "color": ch["Color"].lstrip("#")} for ch in channels
        ]

    def _load_image_data(self, well_id, field_id, channel_id, timepoint_id):
        """Load specific image data"""
        cache_key = (well_id, field_id, channel_id, timepoint_id)
        if cache_key in self._file_cache:
            return self._file_cache[cache_key]

        # Find the file for this combination
        timepoint_info = self.metadata["timepoints"][timepoint_id]
        channel_code = self.metadata["channels_raw"][channel_id]

        filename = f"{well_id}-{field_id + 1}-{channel_code}.tif"
        file_path = timepoint_info["path"] / filename

        # Check if file exists
        if not file_path.exists():
            if self.fill_missing_images:
                # Create a black image with the same dimensions as other images
                sample_info = self._get_sample_image_info()
                black_image = np.zeros((sample_info["height"], sample_info["width"]), 
                                     dtype=sample_info["dtype"])
                self._file_cache[cache_key] = black_image
                print(f"Warning: Missing image file {file_path}, filled with black image")
                return black_image
            else:
                raise FileNotFoundError(f"Image file not found: {file_path}")

        try:
            # Let TiffFile handle the file reading errors naturally
            with tifffile.TiffFile(str(file_path)) as tif:
                page = tif.pages[0]
                data = page.asarray()
                self._file_cache[cache_key] = data
                return data
        except Exception as e:
            if self.fill_missing_images:
                # If file exists but can't be read, also fill with black image
                sample_info = self._get_sample_image_info()
                black_image = np.zeros((sample_info["height"], sample_info["width"]), 
                                     dtype=sample_info["dtype"])
                self._file_cache[cache_key] = black_image
                print(f"Warning: Could not read image file {file_path}: {e}, filled with black image")
                return black_image
            else:
                raise e

    # ImageSource interface methods
    def is_screen(self):
        return self.is_plate

    def get_data(self, well_id, field_id):
        """Get data for a specific well and field"""
        well_id = strip_leading_zeros(well_id)

        if well_id not in self.metadata["wells"]:
            raise ValueError(
                f"Invalid Well: {well_id}. Available: {list(self.metadata['wells'].keys())}"
            )

        field_id = int(field_id)
        if field_id not in self.metadata["fields_raw"]:
            raise ValueError(
                f"Invalid Field: {field_id}. Available: {self.metadata['fields_raw']}"
            )

        # Build 5D array: (t, c, z, y, x)
        nt = len(self.metadata["time_points"])
        nc = self.metadata["num_channels"]
        sample_info = self._get_sample_image_info()
        ny, nx = sample_info["height"], sample_info["width"]
        nz = 1  # Incucyte is typically 2D

        data = np.zeros((nt, nc, nz, ny, nx), dtype=sample_info["dtype"])

        for t in range(nt):
            for c in range(nc):
                image_data = self._load_image_data(well_id, field_id, c, t)
                # Handle different image shapes
                if len(image_data.shape) == 2:
                    data[t, c, 0, :, :] = image_data
                elif len(image_data.shape) == 3 and image_data.shape[0] == 1:
                    data[t, c, 0, :, :] = image_data[0]
                else:
                    # Take first z-plane if 3D
                    data[t, c, 0, :, :] = (
                        image_data[..., 0] if len(image_data.shape) > 2 else image_data
                    )

        return data

    def get_name(self):
        return self.name

    def get_dim_order(self):
        return self.dim_order

    def get_dtype(self):
        return self.dtype

    def get_pixel_size_um(self):
        return self.pixel_size

    def get_position_um(self, well_id=None):
        well = self.metadata["wells"].get(well_id, {})
        well_info = self.metadata["well_info"]
        x = well.get("CoordX", 0) * well_info.get("max_sizex_um", 0)
        y = well.get("CoordY", 0) * well_info.get("max_sizey_um", 0)
        return {"x": x, "y": y}

    def get_channels(self):
        return self.channels

    def get_nchannels(self):
        return max(self.metadata.get("num_channels", 1), 1)

    def get_rows(self):
        return self.rows

    def get_columns(self):
        return self.columns

    def get_wells(self):
        return self.wells

    def get_time_points(self):
        return self.metadata.get("time_points", [])

    def get_fields(self):
        return self.metadata.get("well_info", {}).get("fields", [])

    def get_well_coords_um(self, well_id):
        """Get well coordinates (placeholder - Incucyte doesn't typically have stage coordinates)"""
        return {"x": 0.0, "y": 0.0}

    def get_acquisitions(self):
        """Return acquisition information based on timepoints"""
        acquisitions = []
        for i, tp in enumerate(self.metadata.get("timepoints", [])):
            acq = {
                "id": i,
                "name": f"Timepoint_{tp['timestamp']}",
                "description": f"Incucyte acquisition at {tp['timestamp']}",
                "date_created": tp["datetime"].isoformat()
                if tp["datetime"]
                else tp["timestamp"],
                "date_modified": tp["datetime"].isoformat()
                if tp["datetime"]
                else tp["timestamp"],
            }
            acquisitions.append(acq)
        return acquisitions

    def get_total_data_size(self):
        return self.metadata.get("max_data_size", 0)

    def print_well_matrix(self):
        """Print a visual representation of the plate layout"""
        s = ""
        well_info = self.metadata.get("well_info", {})
        rows = well_info.get("rows", [])
        cols = [int(c) for c in well_info.get("columns", [])]
        used_wells = set(self.metadata.get("wells", {}).keys())

        # Header with column numbers
        header = "   " + "  ".join(f"{col:2d}" for col in cols)
        s += header + "\n"

        # Each row
        for row_letter in rows:
            row_line = f"{row_letter}  "
            for col_num in cols:
                well_id = f"{row_letter}{col_num}"
                row_line += " + " if well_id in used_wells else "   "
            s += row_line + "\n"

        return s

    def print_timepoint_well_matrix(self):
        """Print timepoint vs well matrix"""
        s = ""
        timepoints = self.metadata.get("timepoints", [])
        wells = list(self.metadata.get("wells", {}).keys())

        # Header
        header = "Timepoint   " + "  ".join(f"{well:>3}" for well in wells)
        s += header + "\n"

        # Check which wells have data at each timepoint
        for tp in timepoints:
            line = f"{tp['timestamp']:>9}   "
            for well in wells:
                # Check if any files exist for this well at this timepoint
                has_data = any(
                    (tp["path"] / f"{well}-{field + 1}-{channel}.tif").exists()
                    for field in self.metadata.get("fields_raw", [])
                    for channel in self.metadata.get("channels_raw", [])
                )
                line += " + " if has_data else "   "
            s += line + "\n"

        return s

    def close(self):
        """Clean up resources"""
        self._file_cache.clear()
