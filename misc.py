all_nans = True
d        = t_hc.d.min()
while all_nans:

    in_data = t_hc.where(t_hc.d==d,drop=True)

    if 0 < np.isfinite(
        xr.ufuncs.isfinite(in_data.data).sum(['member','step','time']
        ).values
    ).sum():

        weights =  1 - in_data.d/d
        out.append( (in_data * weights).sum('points') / weights.sum())
        all_nans=False
    else:
        d += 5



    # points = tuple(map(np.array,zip(*hc.points.values)))
    out = []

    for loc in locations:

        nlon = loc.lon.values.item()
        nlat = loc.lat.values.item()

        t_hc = hc.assign_coords(d=distance(nlat, nlon, hc.lat, hc.lon))

        t_hc = t_hc.sortby('d')

        for p in t_hc.points:
            if 0 < np.isfinite(
                xr.ufuncs.isfinite(
                    t_hc.data.sel(points=p)
                ).sum(['member','step','time']).values
            ).sum():

                out.append(t_hc.sel(points=p))
                break

    print(xr.concat(out,locations))
