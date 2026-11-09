"""Keep the original inference implementation; add explicit recovery provenance."""
def transform(source):
    marker="original_runtime='UNKNOWN',current_torch=torch.__version__"
    assert source.count(marker)==1
    return source.replace(marker,"recovery=RECOVERY_PROVENANCE,"+marker)
