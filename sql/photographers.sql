-- One row per photographer that has at least one published image.
-- No names, contact details or national ids: photographer_id is the only identifier.
-- grade is Grade#fa_title (0..6, -1 = "در حال بررسی"/pending, NULL = never graded).
-- signup_at is kept for the temporal train/test split.
SELECT p.id                                                    AS photographer_id,
       g.fa_title::int                                         AS grade,
       p.created_at                                            AS signup_at,
       p.approved,
       p.has_studio,
       p.educated,
       p.video_service,
       p.gender,
       p.city_id,
       (coalesce(p.instagram, '')        <> '')                AS has_instagram,
       (coalesce(p.website, '')          <> '')                AS has_website,
       (coalesce(p.online_portfolio, '') <> '')                AS has_online_portfolio,
       (SELECT count(*) FROM experiences ex WHERE ex.photographer_id = p.id) AS n_experiences,
       pc.n_photos,
       pc.n_expertises
FROM photographers p
JOIN (SELECT e.photographer_id,
             count(ph.id)             AS n_photos,
             count(DISTINCT e.id)     AS n_expertises
      FROM photos ph
      JOIN expertises e ON e.id = ph.expertise_id
      WHERE ph.published AND ph.media_type = 'image'
      GROUP BY e.photographer_id) pc ON pc.photographer_id = p.id
LEFT JOIN grades g ON g.id = p.grade_id
ORDER BY p.id
