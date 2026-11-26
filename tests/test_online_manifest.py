"""online/calib_pilot/MANIFEST.sha256 must describe the files that are actually in the folder.

The manifest is what a deployment is checked against (online/check_pavlovia_deploy.py). If an upload file is edited
and the manifest is not rewritten, the check would compare against a stale list; this test fails first.
Rewrite with: python online/check_pavlovia_deploy.py --write-manifest
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "online"))
import check_pavlovia_deploy as D


def test_manifest_matches_the_upload_files():
    assert D.check_local("calib_pilot") == []


def test_manifest_lists_all_fifteen_files():
    files = D.upload_list("calib_pilot")
    assert len(files) == 15 and set(D.read_manifest("calib_pilot")) == set(files)


def test_describe_names_the_kind_of_difference():
    full = b"line one\nline two\nline three\n"
    assert D.describe(full, full[:12]).startswith("cut short: the first 12 of")
    assert D.describe(full, b"line one\n  line two\n    line three").startswith("same text, different indentation")
    assert D.describe(full, b"line one\nLINE two\nline three\n").startswith("different content from byte 9")
