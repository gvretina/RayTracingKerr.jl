module run

import ..image
import ..angles

using Colors
using Images, ImageView
using StaticArrays

function trace_rays(metric)
    T = Float64 # this is wrong, it should be inferred from the arguments
    ### Initial Conditions ###
    init_r = T(30)
    dist = T(1)
    λ0, λ1 = T(0), T(100)
    tspan = (λ0, λ1)
    pos = T[0, -init_r, 0, 0]

    ####### Background Image ##########
    background, back_angs = image.get_background("eso0932a.jpg")#("eso0932a.jpg")#("a.jpg")

    ####### Rotations ##########
    R_euler = angles.get_rotation_matrix(0, -5, 0)
    # i dont like this!
    pos[2:end] = R_euler * pos[2:end]


    ####### Velocities ########
    h, w = size(background)
    N = 5
    th, ph = angles.get_velocity_angles(90, 120, dist,
        h ÷ N, w ÷ N)

    g = metric(SVector{D,T}(pos))
    gu = inv(g)

    img = Array{RGBA{N0f8},2}(undef, length(th), length(ph))

    tol = 1e-12
    ## rewrite this, this is wrong. use multiple dispatch
    if typeof(ph) == T
        image.update_image!(img,
            [th], [ph],
            g, gu,
            tspan, tol,
            pos, R_euler,
            background, back_angs,
            metric
        )
    else
        image.update_image!(img,
            th, ph,
            g, gu,
            tspan, tol,
            pos, R_euler,
            background, back_angs,
            metric
        )
    end
    #imshow(img)
    #save("win.png",img)
    return img
end

end
