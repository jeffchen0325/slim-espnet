import sys
from tasks.asr.main import main
from espnet3.systems.asr.system import ASRSystem

if __name__ == "__main__":
    main(
        sys_args=sys.argv[1:],
        system_cls=ASRSystem,
    )
