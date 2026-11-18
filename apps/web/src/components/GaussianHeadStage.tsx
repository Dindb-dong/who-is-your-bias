import { useMemo } from 'react';
import { JENSEN_SOURCE_FITTED_AVATAR } from '../avatar/builtin';
import { AvatarStage, type AvatarStageProps } from './AvatarStage';

interface Props extends Omit<AvatarStageProps, 'gaussianHead'> { bundle?: string }

/** Compatibility entry point: the Gaussian face now shares the articulated
 * body renderer, audio clock, speech, attention and performance controls. */
export function GaussianHeadStage({ bundle = 'jensen_a_c1c3_240k_sh',
  modelUrl = JENSEN_SOURCE_FITTED_AVATAR.modelUrl, framing = 'upper_body', ...props }: Props) {
  const gaussianHead = useMemo(() => {
    const override = new URLSearchParams(window.location.search).get('headBundle');
    return { bundle: override && /^[A-Za-z0-9_-]+$/.test(override) ? override : bundle };
  }, [bundle]);
  return <AvatarStage {...props} modelUrl={modelUrl} framing={framing} gaussianHead={gaussianHead}/>;
}

export default GaussianHeadStage;
