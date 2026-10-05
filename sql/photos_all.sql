-- EVERY published image photo of every graded photographer (grade 0..6), no per-photographer cap.
-- Used for the "admins look at the whole portfolio" experiment. expertise_id is kept for per-expertise aggregation.
SELECT e.photographer_id, e.shoot_type_id, e.id AS expertise_id, ph.id AS photo_id, ph.file AS file_name
FROM photos ph
JOIN expertises e ON e.id = ph.expertise_id
JOIN photographers p ON p.id = e.photographer_id
JOIN grades g ON g.id = p.grade_id
WHERE ph.published AND ph.media_type = 'image' AND ph.file IS NOT NULL
  AND g.fa_title::int BETWEEN 0 AND 6
ORDER BY e.photographer_id, ph.id
