-- Profile fields used by the model. Dirty values are NULLed here (not imputed): birthdays outside 1940-2008,
-- years_been_photographer outside 0..60, projects_payed_count outside 0..5000. qrate is the stored value;
-- n_feedback is how many customer feedbacks it is averaged over, needed to normalise it.
-- experiences can have 2 rows per photographer: the latest one is used.
SELECT p.id AS photographer_id,
       CASE WHEN p.birthday BETWEEN '1940-01-01' AND '2008-01-01'
            THEN extract(year FROM now()) - extract(year FROM p.birthday) END       AS age,
       CASE WHEN e.years_been_photographer BETWEEN 0 AND 60 THEN e.years_been_photographer END AS years_been_photographer,
       CASE WHEN e.projects_payed_count BETWEEN 0 AND 5000 THEN e.projects_payed_count END     AS projects_payed_count,
       p.qrate,
       coalesce(f.n, 0) AS n_feedback
FROM photographers p
LEFT JOIN (SELECT DISTINCT ON (photographer_id) * FROM experiences ORDER BY photographer_id, id DESC) e
       ON e.photographer_id = p.id
LEFT JOIN (SELECT pr.photographer_id, count(*) AS n
           FROM feed_backs fb JOIN projects pr ON pr.id = fb.project_id GROUP BY 1) f
       ON f.photographer_id = p.id
ORDER BY p.id
