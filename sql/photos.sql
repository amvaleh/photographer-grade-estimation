-- Up to :per_photographer published image photos per photographer, deterministic and
-- spread across that photographer's expertises (round-robin by md5 order), so a
-- photographer with one huge wedding gallery does not drown out their other work.
-- EXIF: artist, copyright and serial_number are deliberately NOT exported (personal data).
WITH ranked AS (
  SELECT e.photographer_id, e.shoot_type_id, e.id AS expertise_id, ph.id AS photo_id,
         ph.file AS file_name, ph.created_at AS uploaded_at, ph.promote, ph.exif_id,
         row_number() OVER (PARTITION BY e.photographer_id, e.id
                            ORDER BY md5(ph.id::text)) AS rank_in_expertise
  FROM photos ph
  JOIN expertises e ON e.id = ph.expertise_id
  WHERE ph.published AND ph.media_type = 'image' AND ph.file IS NOT NULL),
picked AS (
  SELECT *, row_number() OVER (PARTITION BY photographer_id
                               ORDER BY rank_in_expertise, md5(photo_id::text)) AS pick
  FROM ranked)
SELECT k.photographer_id, k.shoot_type_id, k.expertise_id, k.photo_id, k.file_name,
       k.uploaded_at, k.promote,
       (x.id IS NOT NULL) AS has_exif,
       x.camera_model, x.lens, x.iso, x.f_number, x.exposure_time, x.focal_length,
       x.software, x.exposure, x.contrast, x.highlights, x.shadows, x.whites, x.blacks,
       x.clarity, x.saturation, x.white_balance, x.color_temperature, x.tint
FROM picked k
LEFT JOIN exifs x ON x.id = k.exif_id
WHERE k.pick <= :per_photographer
ORDER BY k.photographer_id, k.pick
