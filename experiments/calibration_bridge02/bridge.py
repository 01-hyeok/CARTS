"""CLI only. Do not import under name 'bridge': upstream uses that module name."""
import sys
from calibration_adapter import main
if __name__=='__main__':sys.exit(main())
