"""Fix parameter mapping in nanobot tau_bench.py tools.

book_reservation: convert simplified date+flight_number to flights array
update_reservation_flights: rename flights → new_flights
"""
import sys
sys.path.insert(0, '/data/wangye/trainable-openclaw')

path = '/data/wangye/trainable-openclaw/nanobot-0.2.1/nanobot/agent/tools/tau_bench.py'
with open(path, 'r') as f:
    content = f.read()

# Fix: add _map_params function before _run, and call it from _run
# First check if _map_params already exists and is mangled
old_mangled = (
    '# ── parameter mapping: nanobot schema → original tau-bench schema ─────────'
    'def _map_params(tool_name: str, kwargs: dict):    '
)

if old_mangled in content:
    # Replace the mangled version
    idx = content.find(old_mangled)
    # Find where the next def _run starts
    run_idx = content.find('\ndef _run', idx)
    if run_idx < 0:
        run_idx = content.find('def _run', idx)

    if run_idx > idx:
        mangled = content[idx:run_idx]
        clean = '''# -- parameter mapping: nanobot schema -> original tau-bench schema ---------
def _map_params(tool_name: str, kwargs: dict):
    """Adapt nanobot tool parameters to match original tau-bench tool signatures."""
    if tool_name == "book_reservation":
        # Convert simplified date+flight_number to flights array
        if "flights" not in kwargs:
            fn = kwargs.pop("flight_number", None)
            dt = kwargs.pop("date", None)
            if fn and dt:
                kwargs["flights"] = [{"flight_number": fn, "date": dt}]
        if "flight_type" not in kwargs:
            n = len(kwargs.get("flights", []))
            kwargs["flight_type"] = "round_trip" if n > 1 else "one_way"
        if "payment" not in kwargs:
            kwargs["payment"] = {"method": "credit_card"}
    elif tool_name == "update_reservation_flights":
        if "new_flights" not in kwargs and "flights" in kwargs:
            kwargs["new_flights"] = kwargs.pop("flights")

'''
        content = content.replace(mangled, clean)
        print('Fixed _map_params formatting')

# Add _map_params call inside _run if not already present
old_run = '    for t in register_tau_bench_tools(scenario):\n        if t.name == tool_name:\n            try:\n                return t.execute(kwargs, state)'
if '_map_params(tool_name, kwargs)' not in content:
    content = content.replace(old_run, '    _map_params(tool_name, kwargs)\n' + old_run)
    print('Added _map_params call in _run')

with open(path, 'w') as f:
    f.write(content)

import py_compile
py_compile.compile(path, doraise=True)
print('Compilation OK')
print('Done')
