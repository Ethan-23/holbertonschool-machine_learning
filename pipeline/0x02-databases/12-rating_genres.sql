-- Thing that I need
SELECT g.name AS name, SUM(r.rate) AS rating FROM tv_genres AS g
       LEFT JOIN tv_show_genres AS t ON g.id = t.genre_id
       LEFT JOIN tv_show_ratings AS r ON t.show_id = r.show_id
       GROUP BY g.name ORDER BY rating DESC;
