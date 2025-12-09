# Converter Overview

This project provides a converter that transforms source image data into OME-Tiff or OME-Zarr format.
When creating a github release, a Docker image is built and published to Docker Hub with corresponding version tags
[here](https://hub.docker.com/r/cellularimagingcf/biomero-converter).

## Architecture

The converter workflow consists of:

- Creating a **Source** reader to access image data and metadata.
- Creating a **Writer** to generate OME output.
- The **Writer** queries the **Source** for metadata and data, then writes the output.

```mermaid
classDiagram
    class Converter:::main {
        convert(source path, output path, format)
    }

    class Source["*Source*"]:::abstract_source {
        get_data()*
        get_pixel_size()*
        ...()*
    }

    class TiffSource:::source {
        ...
        get_data()
        get_pixel_size()
        ...()
    }

    class CustomSource["...Source"]:::source {
        ...
        get_data()
        get_pixel_size()
        ...()
    }

    class Writer["*Writer*"]:::abstract_writer {
        write(output path, Source, ...)*
    }

    class OmeTiffWriter:::writer {
        write(output path, Source, ...)
    }

    class OmeZarrWriter:::writer {
        write(output path, Source, ...)
    }

    Source <|-- TiffSource
    Source <|-- CustomSource
    Writer <|-- OmeTiffWriter
    Writer <|-- OmeZarrWriter
    Source <.. Writer
    Converter ..> Writer

    classDef main fill:#ffeecc,stroke:#775500
    classDef source fill:#e8f5e9,stroke:#1b5e20
    classDef abstract_source fill:#ffffff
    classDef writer fill:#e0f7fa,stroke:#006064
    classDef abstract_writer fill:#ffffff
```
