#!/usr/bin/env python
"""Prepare a meas.dat file for retro recon on Siemens instrument"""
import sys, argparse, logging
from pathlib import Path
from twix.meas import MeasFile


log = logging.getLogger("prep_retrorecon")

def _main(argv=sys.argv):
    #Setup command line parser
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument(
        'meas_file', help="The measurment file we want to prepare"
    )
    arg_parser.add_argument(
        "-d", 
        "--dependency-dir",
        default=None,
        help="Directory to look for missing dependencies in (by default look in same dir as 'meas_file')"
    )
    arg_parser.add_argument("-o", "--out-dir", default=".", help="Directory to save output files to")
    arg_parser.add_argument("-v", "--verbose", action="store_true")
    arg_parser.add_argument("--debug", action="store_true")
    args = arg_parser.parse_args(argv[1:])
    # Setup logging
    root_logger = logging.getLogger("")
    root_logger.setLevel(logging.DEBUG)
    stream_formatter = logging.Formatter("%(name)s %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_formatter)
    handlers = [stream_handler]
    if args.debug:
        for handler in handlers:
            handler.setLevel(logging.DEBUG)
    elif args.verbose:
        stream_handler.setLevel(logging.INFO)
    else:
        stream_handler.setLevel(logging.WARN)
    for handler in handlers:
        root_logger.addHandler(handler)
    # Parse main meas file and determine missing dpes
    meas_file = Path(args.meas_file)
    if args.dependency_dir is None:
        args.dependency_dir = meas_file.parent
    else:
        args.dependency_dir = Path(args.dependency_dir)
    args.out_dir = Path(args.out_dir)
    log.debug("Reading main meas.dat file")
    mf = MeasFile(meas_file)
    missing_uids = mf.get_missing_dep_uids()
    # Find / load any missing deps
    prepend = []
    if missing_uids:
        for missing_uid in missing_uids:
            dep_paths = list(args.dependency_dir.glob(f"meas_MID{missing_uid}_*_FID*.dat"))
            if len(dep_paths) > 1:
                print(f"Multiple potential matches for UID {missing_uid}: {dep_paths}")
                return 1
            elif len(dep_paths) == 0:
                print(f"Unable to find dependent meas file with UID {missing_uid}")
            log.info(f"Injecting dependency: {dep_paths[0]}")
            prepend.append(MeasFile(dep_paths[0]).get_meas())
    # Write out modified file
    out_path = args.out_dir / meas_file.name
    log.info(f"Saving output to: {out_path}")
    mf.save(out_path, prepend)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
