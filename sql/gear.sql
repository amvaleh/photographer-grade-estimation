-- Self-declared camera bodies and lenses, one row per (photographer, item). No serial numbers.
SELECT q.photographer_id, 'camera' AS kind, c.brand, c.model
FROM equipments q JOIN equip_cameras ec ON ec.equipment_id = q.id JOIN cameras c ON c.id = ec.camera_id
UNION ALL
SELECT q.photographer_id, 'lens', l.brand, l.model
FROM equipments q JOIN equip_lenzs el ON el.equipment_id = q.id JOIN lenzs l ON l.id = el.lenz_id
ORDER BY 1, 2, 3, 4
