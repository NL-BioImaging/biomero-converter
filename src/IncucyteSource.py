import chardet
from datetime import datetime
import numpy as np
from pathlib import Path
import re
import tifffile
import zipfile

from src.color_conversion import hexrgb_to_rgba
from src.ImageSource import ImageSource
from src.TiffSource import TiffSource
from src.util import strip_leading_zeros, redimension_data


class IncucyteSource(ImageSource):
    """
    ImageSource implementation for Incucyte data

    Handles the specific directory structure:
    EssenFiles/ScanData/YYMM/DD/HHMM/XXXX/*.tif

    Filenames follow pattern: WELL-FIELD-CHANNEL.tif
    e.g., A1-1-C1.tif, B2-1-Ph.tif
    
    Note: Multiple plates can exist in the same archive, identified by the XXXX folder.
    Use plate_id parameter to select a specific plate, or use get_available_plates() 
    to discover all plates in the archive.
    """

    DIAG_ZIP_FILENAME = "Diag.zip"
    DIAG_LOG_FILENAME = "Diag.log"

    def __init__(self, uri, metadata={}, plate_id=None):
        """
        Initialize IncucyteSource.
        
        Args:
            uri (str): Path to the Incucyte archive folder
            metadata (dict): Optional metadata dictionary
            plate_id (str, optional): Specific plate ID to process (e.g., '700', '701').
                                     If None, will use the first available plate or all 
                                     if only one exists.
        """
        super().__init__(uri, metadata)
        self.base_path = Path(self.uri)
        self.scan_data_path = self.base_path / "EssenFiles" / "ScanData"
        self._file_cache = {}
        self._file_caching = False
        self._diag_metadata_cache = None  # Cache for Diag.log parsing
        self._sample_image_info_cache = None  # Cache for sample image info
        # Default to True for filling missing images
        self.fill_missing_images = True
        self.plate_id = plate_id
    
    @staticmethod
    def get_available_plates(uri):
        """
        Discover all available plate IDs in an Incucyte archive.
        
        Args:
            uri (str): Path to the Incucyte archive folder
            
        Returns:
            list: List of plate IDs (strings) found in the archive
        """
        base_path = Path(uri)
        scan_data_path = base_path / "EssenFiles" / "ScanData"
        
        if not scan_data_path.exists():
            raise ValueError(f"Scan data path not found: {scan_data_path}")
        
        plate_ids = set()
        
        # Navigate through the directory structure to find all plate IDs
        for year_month in scan_data_path.iterdir():
            if not year_month.is_dir():
                continue
            for day in year_month.iterdir():
                if not day.is_dir():
                    continue
                for time_dir in day.iterdir():
                    if not time_dir.is_dir():
                        continue
                    for plate_dir in time_dir.iterdir():
                        if plate_dir.is_dir():
                            plate_ids.add(plate_dir.name)
        
        return sorted(list(plate_ids))

    def enable_file_caching(self, file_caching=True):
        """
        Enable or disable file caching for image data.

        Args:
            file_caching (bool): If True, enable file caching; if False, disable it.
        """
        self._file_caching = file_caching
        if not file_caching:
            self._file_cache.clear()

    def _parse_diag_log(self, diag_zip_path):
        """
        Parse Diag.log from a Diag.zip file to extract imaging metadata.
        
        Args:
            diag_zip_path (Path): Path to Diag.zip file
            
        Returns:
            dict: Dictionary with 'pixel_sizes' (dict of mag->size),
                  'experiments' (dict of expid->metadata), or None if failed
        """
        try:
            with zipfile.ZipFile(diag_zip_path) as zip_ref:
                if self.DIAG_LOG_FILENAME not in zip_ref.namelist():
                    return None

                raw = zip_ref.read(self.DIAG_LOG_FILENAME)
                detection = chardet.detect(raw)
                content = raw.decode(detection['encoding'], errors='ignore')

                # Parse imaging specifications
                pixel_sizes = {}
                mag_pattern = r'(\d+)x:\s+.*?Image Resolution:\s+([\d.]+)\s+microns/pixel'
                for match in re.finditer(mag_pattern, content, re.DOTALL):
                    mag = match.group(1) + 'x'
                    pixel_size = float(match.group(2))
                    pixel_sizes[mag] = pixel_size

                # Parse experiment entries
                experiments = {}
                # Match ExpID and capture next 2 lines for Lmp info
                exp_pattern = r'ExpID=(\d+)[^\n]*Mag=(\d+x)[^\n]*(?:\n[^\n]*)?'
                for match in re.finditer(exp_pattern, content):
                    exp_id = match.group(1)
                    mag = match.group(2)

                    # Extract all exposure times from matched section
                    exp_section = match.group(0)
                    acq_times = re.findall(r'AcqTime=(\d+)', exp_section)

                    experiments[exp_id] = {
                        'magnification': mag,
                        'exposure_times_ms': [int(t) for t in acq_times] if acq_times else None,
                        'pixel_size_um': pixel_sizes.get(mag)
                    }

                return {
                    'pixel_sizes': pixel_sizes,
                    'experiments': experiments
                }
        except Exception as e:
            print(f"Warning: Could not parse {self.DIAG_LOG_FILENAME} from {diag_zip_path}: {e}")
            return None

    def _find_and_parse_diag_log(self):
        """
        Find the first Diag.zip in the scan data and parse it.
        Caches the result to avoid repeated parsing.
        
        Returns:
            dict: Parsed diag metadata or None if not found
        """
        # Return cached result if available
        if self._diag_metadata_cache is not None:
            return self._diag_metadata_cache
        
        # Look for first Diag.zip in the scan data
        diag_zip_files = list(self.scan_data_path.rglob(self.DIAG_ZIP_FILENAME))
        
        if diag_zip_files:
            self._diag_metadata_cache = self._parse_diag_log(diag_zip_files[0])
        else:
            self._diag_metadata_cache = {}  # Cache empty dict to avoid re-search
        
        return self._diag_metadata_cache if self._diag_metadata_cache else None

    def init_metadata(self):
        """Initialize all metadata from Incucyte structure"""
        self._scan_timepoints()  # Must be first to set plate_id
        self._get_experiment_metadata()  # Uses plate_id in name
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
        self.scales = [1]

        nt = len(self.metadata["time_points"])
        nc = self.metadata["num_channels"]
        sample_info = self._get_sample_image_info()
        ny, nx = sample_info["height"], sample_info["width"]
        nz = 1  # Incucyte is typically 2D
        self.shape = nt, nc, nz, ny, nx

        return self.metadata

    def _get_experiment_metadata(self):
        """Extract experiment metadata from folder structure"""
        experiment_name = self.base_path.name
        
        # Add plate ID to name (plate_id is set by _scan_timepoints)
        if self.plate_id:
            experiment_name = f"{experiment_name}_plate{self.plate_id}"
        
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
        found_plate_ids = set()

        print(f"Scanning directory: {self.scan_data_path}")

        if not self.scan_data_path.exists():
            raise ValueError(
                f"Scan data path not found: {self.scan_data_path}"
            )

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
                    # Navigate through plate ID directories (XXXX)
                    for plate_dir in time_dir.iterdir():
                        if not plate_dir.is_dir():
                            continue
                        
                        current_plate_id = plate_dir.name
                        found_plate_ids.add(current_plate_id)
                        
                        # Filter by plate_id if specified
                        if self.plate_id is not None:
                            if current_plate_id != self.plate_id:
                                continue
                        
                        timepoint_path = plate_dir
                        timestamp = (
                            f"{year_month.name}_{day.name}_{time_dir.name}"
                        )

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
                            "plate_id": current_plate_id,
                        }
                        timepoints.append(timepoint_info)

                        # Scan TIFF files in this timepoint
                        for tiff_file in timepoint_path.glob("*.tif"):
                            well, field, channel = self._parse_filename(tiff_file.name)
                            if well and field is not None and channel:
                                wells.add(well)
                                fields.add(field)
                                channels.add(channel)

        # Handle plate selection
        if self.plate_id is None:
            # Auto-select plate
            if len(found_plate_ids) == 0:
                raise ValueError("No plates found in the archive")
            elif len(found_plate_ids) == 1:
                # Single plate - use it automatically
                self.plate_id = list(found_plate_ids)[0]
            else:
                # Multiple plates - use first with warning
                plate_list = ", ".join(sorted(found_plate_ids))
                print(
                    f"Warning: Multiple plates found ({plate_list}). "
                    f"Using first plate: {sorted(found_plate_ids)[0]}"
                )
                print(
                    "To process a specific plate, use: "
                    "IncucyteSource(uri, plate_id='XXX')"
                )
                print(
                    "To process all plates, call get_available_plates() "
                    "and create separate sources"
                )
                self.plate_id = sorted(found_plate_ids)[0]
            
            # Filter timepoints to selected plate
            timepoints = [
                tp for tp in timepoints if tp["plate_id"] == self.plate_id
            ]
        else:
            # Validate specified plate_id
            if self.plate_id not in found_plate_ids:
                raise ValueError(
                    f"Plate ID '{self.plate_id}' not found. "
                    f"Available plates: {', '.join(sorted(found_plate_ids))}"
                )
            # Filter timepoints to specified plate
            timepoints = [
                tp for tp in timepoints if tp["plate_id"] == self.plate_id
            ]
        
        # Store found plate IDs in metadata
        self.metadata["available_plates"] = sorted(found_plate_ids)
        self.metadata["selected_plate"] = self.plate_id

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

        plate_info = (
            f" (plate: {self.plate_id})" if self.plate_id else ""
        )
        print(
            f"Found{plate_info}: {len(timepoints)} timepoints, "
            f"{len(wells)} wells, {len(fields)} fields, "
            f"{len(channels)} channels"
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
        
        # Add optional imaging metadata if available
        if "magnification" in sample_image_info:
            well_info["Magnification"] = sample_image_info["magnification"]
        if "exposure_times_ms" in sample_image_info:
            well_info["ExposureTimes_ms"] = sample_image_info["exposure_times_ms"]

        self.metadata.update({"wells": wells_dict, "well_info": well_info})

    def _get_sample_image_info(self):
        """Get image dimensions and bit depth from first available TIFF.
        Attempts to get accurate pixel size from Diag.log if available.
        Caches the result to avoid repeated parsing."""
        
        # Return cached result if available
        if self._sample_image_info_cache is not None:
            return self._sample_image_info_cache
        
        # Try to get calibrated pixel size from Diag.log
        diag_metadata = None
        pixel_size_from_diag = None
        magnification = None
        exposure_time = None
        
        if self.plate_id:
            diag_metadata = self._find_and_parse_diag_log()
            if diag_metadata and 'experiments' in diag_metadata:
                exp_info = diag_metadata['experiments'].get(self.plate_id)
                if exp_info:
                    pixel_size_from_diag = exp_info.get('pixel_size_um')
                    magnification = exp_info.get('magnification')
                    exposure_times = exp_info.get('exposure_times_ms')
                    # Use the exposure times list if available
                    exposure_time = exposure_times
                    if pixel_size_from_diag:
                        print(f"Found calibrated pixel size from {self.DIAG_LOG_FILENAME}: "
                              f"{pixel_size_from_diag} µm/pixel "
                              f"(Magnification: {magnification})")
        
        for timepoint in self.metadata["timepoints"]:
            for tiff_file in timepoint["path"].glob("*.tif"):
                try:
                    # Get actual image dimensions from the file
                    with tifffile.TiffFile(str(tiff_file)) as tif:
                        page = tif.pages.first
                        width = page.sizes["width"]
                        height = page.sizes["height"]
                        dtype = page.dtype
                        bits = dtype.itemsize * 8

                    # Use calibrated pixel size from Diag.log if available
                    if pixel_size_from_diag:
                        pixel_x = pixel_size_from_diag
                        pixel_y = pixel_size_from_diag
                    else:
                        # Fallback to TIFF metadata
                        temp_tiff_source = TiffSource(str(tiff_file))
                        temp_tiff_source.init_metadata()
                        pixel_size = temp_tiff_source.get_pixel_size_um()
                        temp_tiff_source.close()
                        pixel_x = pixel_size.get("x")
                        pixel_y = pixel_size.get("y")

                    result = {
                        "width": width,
                        "height": height,
                        "bits": bits,
                        "dtype": dtype,
                        "pixel_x": pixel_x,
                        "pixel_y": pixel_y,
                    }
                    
                    # Add optional metadata if available
                    if magnification:
                        result["magnification"] = magnification
                    if exposure_time:
                        result["exposure_times_ms"] = exposure_time
                    
                    # Cache the result
                    self._sample_image_info_cache = result
                    return result
                    
                except Exception as e:
                    print(f"Could not read sample image {tiff_file}: {e}")
                    continue

        # If no valid TIFF files found
        raise ValueError(
            f"No valid TIFF files found in experiment directory: "
            f"{self.scan_data_path}"
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
            {"label": channel["Dye"], "color": hexrgb_to_rgba(channel["Color"].lstrip("#"))} for channel in channels
        ]

    def _load_image_data(self, well_id, field_id, channel_id, timepoint_id, level=0):
        """Load specific image data"""
        cache_key = (well_id, field_id, channel_id, timepoint_id, level)
        if cache_key in self._file_cache:
            return self._file_cache[cache_key]

        data = None

        # Find the file for this combination
        timepoint_info = self.metadata["timepoints"][timepoint_id]
        channel_code = self.metadata["channels_raw"][channel_id]

        filename = f"{well_id}-{field_id + 1}-{channel_code}.tif"
        file_path = timepoint_info["path"] / filename

        message = ""
        # Check if file exists
        if not file_path.exists():
            if self.fill_missing_images:
                message = f"Warning: Missing image file {file_path}, filled with black image"
            else:
                raise FileNotFoundError(f"Image file not found: {file_path}")

        try:
            # Let TiffFile handle the file reading errors naturally
            with tifffile.TiffFile(str(file_path)) as tif:
                data = tif.asarray(level=level)
        except Exception as e:
            if self.fill_missing_images:
                message = f"Warning: Could not read image file {file_path}: {e}, filled with black image"
            else:
                raise e

        if data is None and self.fill_missing_images:
            # Create a black image with the same dimensions as other images
            sample_info = self._get_sample_image_info()
            data = np.zeros((sample_info["height"], sample_info["width"]), dtype=sample_info["dtype"])
            print(message)

        if self._file_caching:
            self._file_cache[cache_key] = data
        return data

    # ImageSource interface methods
    def is_screen(self):
        return self.is_plate

    def get_data(self, dim_order, level=0, well_id=None, field_id=None, **kwargs):
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

        data = np.zeros(self.shape, dtype=sample_info["dtype"])

        for t in range(nt):
            for c in range(nc):
                image_data = self._load_image_data(well_id, field_id, c, t, level=level)
                # Handle different image shapes
                if len(image_data.shape) == 2:
                    data[t, c, 0, :, :] = image_data
                elif len(image_data.shape) == 3 and image_data.shape[0] == 1:
                    data[t, c, 0, :, :] = image_data[0]
                else:
                    # Take first z-plane if 3D TODO handle 3D if needed, but unlikely for Incucyte
                    data[t, c, 0, :, :] = (
                        image_data[..., 0] if len(image_data.shape) > 2 else image_data
                    )

        return redimension_data(data, self.dim_order, dim_order)

    def get_shape(self):
        return self.shape

    def get_scales(self):
        return self.scales

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

    def is_rgb(self):
        """
        Check if the source is a RGB(A) image.
        Incucyte data stores channels separately, not as RGB.
        """
        return False

    def close(self):
        """Clean up resources"""
        self._file_cache.clear()
