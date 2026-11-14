import io
import sys
from pathlib import Path
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fetch_minimal_assets import OffsetBuffer


def test_buffer_retains_absolute_zip_offsets():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("unwanted", b"abc" * 1000)
        archive.writestr("wanted", b"payload" * 100)
    raw = buffer.getvalue()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        member = archive.getinfo("wanted")
        archive.fp = OffsetBuffer(raw[member.header_offset :], member.header_offset)
        assert archive.read(member) == b"payload" * 100
