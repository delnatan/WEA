#!/usr/bin/env python

# make sure you're in the right python environment before running the program
# 05/16/2022 - you can now make a symlink of this file so that the flask
#              app will run

import argparse
import binascii
import os
import shutil
from pathlib import Path

from flask import Flask, render_template, request, send_from_directory

__module_file = Path(__file__).resolve()
__module_dir = __module_file.parent

FlaskApp = Flask(
    "Flask Image Gallery",
    static_folder=str(__module_dir / "static"),
    template_folder=str(__module_dir / "templates"),
)
FlaskApp.config["IMAGE_EXTS"] = [".png", ".jpg", ".jpeg", ".gif", ".tiff", ".tif"]


def encode(x):
    return binascii.hexlify(x.encode("utf-8")).decode()


def decode(x):
    return binascii.unhexlify(x.encode("utf-8")).decode()


def get_filename(s):
    return os.path.basename(decode(s))


@FlaskApp.route("/", methods=["POST", "GET"])
def home():
    if request.method == "GET":
        # GET, when page is initially loaded
        disp_dir = FlaskApp.config["DISPLAY_DIR"]
        image_paths = []

        # populate image path for displaying cellpose results
        for file in os.listdir(disp_dir):
            valid = os.path.isfile(
                os.path.join(disp_dir, file)
            ) and not file.startswith(".")
            if valid:
                if any(file.endswith(ext) for ext in FlaskApp.config["IMAGE_EXTS"]):
                    image_paths.append(encode(os.path.join(disp_dir, file)))

        return render_template(
            "index.html",
            paths=image_paths,
            get_filename=get_filename,
            Nimages=len(image_paths),
        )

    elif request.method == "POST":
        # POST, when form is submitted

        disp_dir = FlaskApp.config["DISPLAY_DIR"]
        # get a list of files that have been marked
        markedfiles = request.form.getlist("marked_files")
        # and move them to a new folder
        moved_disp_dir = os.path.join(disp_dir, "marked")

        if not os.path.exists(moved_disp_dir):
            os.mkdir(moved_disp_dir)

        # moved marked files into "marked" subfolder
        for f in markedfiles:
            shutil.move(os.path.join(disp_dir, f), os.path.join(moved_disp_dir, f))

        # repopulate page with the remaining images
        image_paths = []

        for file in os.listdir(disp_dir):
            valid = os.path.isfile(
                os.path.join(disp_dir, file)
            ) and not file.startswith(".")
            if valid:
                if any(file.endswith(ext) for ext in FlaskApp.config["IMAGE_EXTS"]):
                    image_paths.append(encode(os.path.join(disp_dir, file)))

        return render_template(
            "index.html",
            paths=image_paths,
            get_filename=get_filename,
            Nimages=len(image_paths),
        )


@FlaskApp.route("/cdn/<path:filepath>")
def download_file(filepath):
    dir, filename = os.path.split(decode(filepath))
    return send_from_directory(dir, filename, as_attachment=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Usage: %prog [options]")
    parser.add_argument("disp_dir", help="Cellpose results (boundaries marked) path")
    parser.add_argument(
        "-l",
        "--listen",
        dest="host",
        default="127.0.0.1",
        help="address to listen on [127.0.0.1]",
    )
    parser.add_argument(
        "-p",
        "--port",
        metavar="PORT",
        dest="port",
        type=int,
        default=5000,
        help="port to listen on [5000]",
    )
    args = parser.parse_args()

    FlaskApp.config["DISPLAY_DIR"] = args.disp_dir
    # disable caching
    FlaskApp.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    FlaskApp.run(host=args.host, port=args.port, debug=True)
