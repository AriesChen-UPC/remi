function [x, y] = selectDataPoints(~, ax)
    roi = drawpoint(ax);
    x = roi.Position(1);
    y = roi.Position(2);
end