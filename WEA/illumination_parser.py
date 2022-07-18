import re
import yaml

def fetch_full_hdr_string(reader):
    return reader.parser._raw_metadata.image_text_info[b"SLxImageTextInfo"][
        b"TextInfoItem_5"
    ].decode()


def fetch_camera_hdr(reader):
    hdrstr = reader.parser._raw_metadata.image_text_info[b"SLxImageTextInfo"][
        b"TextInfoItem_6"
    ].decode()
    # remove title and rejoin metadata text
    hdr_rem_str = "\n".join(hdrstr.split("\r\n")[1:])
    return yaml.safe_load(hdr_rem_str)


def get_exposures(reader):
    camhdr = fetch_camera_hdr(reader)
    exposures = []
    for key, value in camhdr.items():
        exposures.append(value["Exposure"].replace(" ", "").strip("ms"))
    return exposures


def get_illuminator_voltage(reader):
    ptn = r"Illuminator\(Sola\) Voltage: (\d+\.\d)"
    longhdr = fetch_full_hdr_string(reader)
    voltages = re.findall(ptn, longhdr)
    return voltages


def get_illumination_info(reader):
    return list(
        zip(
            reader.metadata["channels"],
            get_exposures(reader),
            get_illuminator_voltage(reader),
        )
    )